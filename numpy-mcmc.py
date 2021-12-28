from typing import Callable, Tuple, Dict, Any, List

import jax.numpy as np

# type aliases
Array = Any 
States = Dict[Tuple[int,int], Array]  # Indexed by tuples (batch index, node index)

class NumpyMCTS():
    """MCTS for one token"""

    def __init__(self, root_fun: Callable[[], Tuple[Array, Any, States]], rec_fun: Callable[[List, int], Tuple], batch_size: int, num_simulations, num_actions, num_sparse_actions, pb_c_init):
        self._batch_size = batch_size
        self._num_simulations = num_simulations
        self._num_actions = num_actions
        self._num_sparse_actions = min(num_sparse_actions, num_actions)
        self._pb_c_init = pb_c_init

        self._root_fun = root_fun  # a function called at the root
        self._rec_fun = rec_fun  # a function called in the tree
        self._adaptive_min_values = np.zeros(batch_size, dtype=np.float32)
        self._adaptive_max_values = np.zeros(batch_size, dtype=np.float32)

        # Allocate all necessary storage.
        # For a given search associated to a batch-index, node i is the i-th node
        # to be expanded. Node 0 corresponds to the root node.
        num_nodes: int = num_simulations + 1
        batch_node = (batch_size, num_nodes)
        self._num_nodes = num_nodes
        self._visit_counts = np.zeros(batch_node, dtype=np.int32)
        self._values = np.zeros(batch_node, dtype=np.float32)
        self._raw_values = np.zeros(batch_node, dtype=np.float32)
        self._parents = np.zeros(batch_node, dtype=np.int32)
        # action_from_parents[b, i] is the action taken to reach node i.
        # Note that action_from_parents[b, 0] will remain -1, as we do not know,
        # when doing search from the root, what action led to the root.
        self._action_from_parents = np.zeros(batch_node, dtype=np.int32)
        # The 0-indexed depth of the node. The root is the only 0-depth node.
        # The depth of node i, is the depth of its parent + 1.
        self._depth = np.zeros(batch_node, dtype=np.int32)
        self._is_terminal = np.full(batch_node, False, dtype=np.bool)

        # To avoid costly numpy ops, we store a sparse version of the actions.
        # We select the top k actions according to the policy, and keep a mapping
        # of indices from 0 to k-1 to the actual action indices in the
        # self._topk_mapping tensor.
        batch_node_action = (batch_size, num_nodes, self._num_sparse_actions)
        self._topk_mapping = np.zeros(batch_node_action, dtype=np.int32)
        self._children_index = np.zeros(batch_node_action, dtype=np.int32)
        self._children_prior = np.zeros(batch_node_action, dtype=np.float32)
        self._children_values = np.zeros(batch_node_action, dtype=np.float32)
        self._children_visits = np.zeros(batch_node_action, dtype=np.int32)
        self._states: States = {}
        self._batch_range = np.arange(batch_size)
        self._reset_tree()

    def _reset_tree(self):
        """Resets the tree arrays."""
        self._visit_counts.fill(0)
        self._values.fill(0)
        self._parents.fill(-1)
        self._action_from_parents.fill(-1)
        self._depth.fill(0)

        self._topk_mapping.fill(-1)
        self._children_index.fill(-1)
        self._children_prior.fill(0.0)
        self._children_values.fill(0.0)
        self._children_visits.fill(0)
        self._states: States = {}  # Indexed by tuples (batch index, node index)

    def search(self, raw_states):
        self._reset_tree()
        # Evaluate the root.
        prior, values, states = self._root_fun(raw_states)

        self._adaptive_min_values = values
        self._adaptive_max_values = values + 1e-6

        root_index = 0
        self.create_node(root_index, prior, values, states,
                          np.full(self._batch_size, False, dtype=np.bool))

        # Do simulations, expansions, and backwards.
        leaf_indices = np.zeros((self._batch_size), np.int32)
        for sim in range(self._num_simulations):
            node_indices, actions = self.simulate()
            next_node_index = sim + 1  # root is 0, therefore we offset by 1.
            self.expand(node_indices, actions, next_node_index)
            leaf_indices.fill(next_node_index)
            self.backward(leaf_indices)

        return self.dense_visit_counts()

    def dense_visit_counts(self):
        root_index = 0
        root_visit_counts = self._children_visits[:, root_index, :]
        dense_visit_counts = np.zeros((self._batch_size, self._num_actions))
        dense_visit_counts[self._batch_range[:, None],
                           self._topk_mapping[:, root_index, :]] = root_visit_counts
        return dense_visit_counts

    def simulate(self):
        """Goes down until all elements have reached unexplored actions."""
        node_indices = np.zeros((self._batch_size), np.int32)
        depth = 0
        while True:
            depth += 1
            actions = self.uct_select_action(node_indices)
            next_node_indices = self._children_index[self._batch_range,
                                                     node_indices, actions]
            is_unexplored = next_node_indices == -1
            if is_unexplored.all():
                return node_indices, actions
            else:
                node_indices = np.where(
                    is_unexplored, node_indices, next_node_indices)

    def uct_select_action(self, node_indices):
        """Returns the action selected for a batch of node indices of shape (B)."""
        node_children_prior = self._children_prior[self._batch_range,
                                                   node_indices, :]  # (B, A)
        # (B, A)
        node_children_values = self._children_values[self._batch_range,
                                                     node_indices, :]
        # (B, A)
        node_children_visits = self._children_visits[self._batch_range,
                                                     node_indices, :]
        # (B)
        node_visits = self._visit_counts[self._batch_range, node_indices]

        node_policy_score = np.sqrt(
            node_visits[:, None]) * self._pb_c_init * node_children_prior / (node_children_visits + 1)  # (B, A)

        # Remap values between 0 and 1.
        node_value_score = node_children_values
        node_value_score = (node_value_score != 0) * node_value_score + \
            (node_value_score == 0) * self._adaptive_min_values[:, None]
        node_value_score = (node_value_score - self._adaptive_min_values[:, None]) / (self._adaptive_max_values[:, None]
                                                                                      - self._adaptive_min_values[:, None])

        node_uct_score = node_value_score + node_policy_score  # (B, A)
        actions = np.argmax(node_uct_score, axis=1)
        return actions

    def expand(self, node_indices, actions, next_node_index):
        """Creates and evaluate child nodes from given nodes and unexplored actions."""

        # Retrieve states for nodes to be evaluated.
        states = [self._states[(b, n)] for b, n in enumerate(node_indices)]
        # (B)
        previous_node_is_terminal = self._is_terminal[self._batch_range,
                                                      node_indices[self._batch_range]]

        # Convert sparse actions to dense actions for network computation
        dense_actions = self._topk_mapping[self._batch_range,
                                           node_indices, actions]

        # Evaluate nodes.
        (prior, values, next_states, expanded_node_is_terminal) = self._rec_fun(
            states, dense_actions)

        # Create the new nodes.
        self.create_node(next_node_index, prior, values,
                         next_states, expanded_node_is_terminal)

        # Update the min and max values arrays
        self._adaptive_min_values = np.minimum(
            self._adaptive_min_values, values)
        self._adaptive_max_values = np.maximum(
            self._adaptive_max_values, values)

        # Update tree topology.
        self._children_index[self._batch_range,
                             node_indices, actions] = next_node_index
        self._parents[:, next_node_index] = node_indices
        self._action_from_parents[:, next_node_index] = actions
        self._depth[:, next_node_index] = self._depth[self._batch_range,
                                                      node_indices] + 1

    def create_node(self, node_index: int, prior: Array, values, next_states: List[Array], expanded_node_is_terminal):
        # Truncate the prior to only keep the top k logits
        prior_topk_indices = np.argpartition(
            prior, -self._num_sparse_actions, axis=-1)[:, -self._num_sparse_actions:]
        prior = prior[self._batch_range[:, None], prior_topk_indices]  # (B, A)

        # Store the indices of the top k logits
        self._topk_mapping[self._batch_range,
                           node_index, :] = prior_topk_indices

        # Update prior, values and visit counts.
        self._children_prior[:, node_index, :] = prior

        self._values[:, node_index] = values
        self._raw_values[:, node_index] = values
        self._visit_counts[:, node_index] = 1
        self._is_terminal[:, node_index] = expanded_node_is_terminal

        # Update states.
        for b, next_state in enumerate(next_states):
            self._states[(b, node_index)] = next_state

    def backward(self, leaf_indices):
        """Goes up and updates the tree until all nodes reached the root."""
        node_indices = leaf_indices  # (B)
        leaf_values = self._values[self._batch_range, leaf_indices]
        while True:
            is_root = node_indices == 0
            if is_root.all():
                return
            parents = np.where(
                is_root, 0, self._parents[self._batch_range, node_indices])
            root_mask = 1.0 * is_root
            not_root_mask_int = (1 - is_root)
            not_root_mask = 1.0 - root_mask
            # Update the parent nodes iff their child is not the root.
            # We therefore mask the updates using not_root_mask and root_mask.
            self._values[self._batch_range, parents] = not_root_mask * (self._values[self._batch_range, parents] *
                                                                        self._visit_counts[self._batch_range, parents] + leaf_values) / (self._visit_counts[self._batch_range,
                                                                                                                                                            parents] + 1.0) + root_mask * self._values[self._batch_range, parents]
            self._visit_counts[self._batch_range, parents] += not_root_mask_int
            actions = np.where(
                is_root, 0, self._action_from_parents[self._batch_range, node_indices])
            self._children_values[self._batch_range, parents, actions] = not_root_mask * self._values[self._batch_range,
                                                                                                      node_indices] + root_mask * self._children_values[self._batch_range, parents, actions]
            self._children_visits[self._batch_range,
                                  parents, actions] += not_root_mask_int

            # Go up
            node_indices = parents
