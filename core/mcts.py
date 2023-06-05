import torch

import numpy as np
import core.ctree.cytree as tree

from torch.cuda.amp import autocast as autocast


class MCTS(object):
    def __init__(self, config):
        self.config = config

    def search(self, roots, model, hidden_state_roots, reward_hidden_roots):
        """Do MCTS for the roots (a batch of root nodes in parallel). Parallel in model inference
        Parameters
        ----------
        roots: Any
            a batch of expanded root nodes
        hidden_state_roots: list
            the hidden states of the roots
        reward_hidden_roots: list
            the value prefix hidden states in LSTM of the roots
        """
        with torch.no_grad():
            model.eval()

            # preparation
            num = roots.num
            device = self.config.device
            pb_c_base, pb_c_init, discount = self.config.pb_c_base, self.config.pb_c_init, self.config.discount
            # the data storage of hidden states: storing the states of all the tree nodes
            hidden_state_pool = [hidden_state_roots]
            # 1 x batch x 64
            # the data storage of value prefix hidden states in LSTM
            reward_hidden_c_pool = [reward_hidden_roots[0]]
            reward_hidden_h_pool = [reward_hidden_roots[1]]
            
            ############################
            policy_softmax_pool = []
            np.set_printoptions(precision=6)
            ############################

            # the index of each layer in the tree
            hidden_state_index_x = 0
            # minimax value storage
            min_max_stats_lst = tree.MinMaxStatsList(num)
            min_max_stats_lst.set_delta(self.config.value_delta_max)
            horizons = self.config.lstm_horizon_len

            results = tree.ResultsWrapper(num)

            ### For state similarity #####
            def normalize(x, axis=-1, eps=1e-5):
                norm = np.linalg.norm(x, axis=axis, keepdims=True)
                return x / (norm + eps)

            def cosine_similarity(batch_size, num_prev_compares, past_states, curr_state):
                past_states = past_states.reshape((batch_size, num_prev_compares, -1))
                curr_state = curr_state.reshape((batch_size, 1, -1))
                past_states = normalize(past_states, axis=-1)
                curr_state = normalize(curr_state, axis=-1)
                return (past_states * curr_state).sum(axis=-1)
            ################################

            # total_similar_states_found = 0

            for index_simulation in range(self.config.num_simulations):
                hidden_states = []
                hidden_states_c_reward = []
                hidden_states_h_reward = []

                # prepare a result wrapper to transport results between python and c++ parts
                # traverse to select actions for each root
                # hidden_state_index_x_lst: the first index of leaf node states in hidden_state_pool
                # hidden_state_index_y_lst: the second index of leaf node states in hidden_state_pool
                # the hidden state of the leaf node is hidden_state_pool[x, y]; value prefix states are the same
                hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions = tree.batch_traverse(roots, pb_c_base, pb_c_init, discount, min_max_stats_lst, results)
                # obtain the search horizon for leaf nodes
                search_lens = results.get_search_len()
                # input(f"len of hidden_state_index_x_lst: {len(hidden_state_index_x_lst)}")

                # obtain the states for leaf nodes
                for ix, iy in zip(hidden_state_index_x_lst, hidden_state_index_y_lst):
                    hidden_states.append(hidden_state_pool[ix][iy])
                    hidden_states_c_reward.append(reward_hidden_c_pool[ix][0][iy])
                    hidden_states_h_reward.append(reward_hidden_h_pool[ix][0][iy])

                hidden_states = torch.from_numpy(np.asarray(hidden_states)).to(device).float()
                hidden_states_c_reward = torch.from_numpy(np.asarray(hidden_states_c_reward)).to(device).unsqueeze(0)
                hidden_states_h_reward = torch.from_numpy(np.asarray(hidden_states_h_reward)).to(device).unsqueeze(0)

                last_actions = torch.from_numpy(np.asarray(last_actions)).to(device).unsqueeze(1).long()

                # evaluation for leaf nodes
                if self.config.amp_type == 'torch_amp':
                    with autocast():
                        network_output = model.recurrent_inference(hidden_states, (hidden_states_c_reward, hidden_states_h_reward), last_actions)
                else:
                    network_output = model.recurrent_inference(hidden_states, (hidden_states_c_reward, hidden_states_h_reward), last_actions)

                hidden_state_nodes = network_output.hidden_state
                value_prefix_pool = network_output.value_prefix.reshape(-1).tolist()
                value_pool = network_output.value.reshape(-1).tolist()
                policy_logits_pool = network_output.policy_logits.tolist()
                reward_hidden_nodes = network_output.reward_hidden

                ############################
                def softmax(matrix):
                    e_x = np.exp(matrix - np.max(matrix, axis=1, keepdims=True))
                    return e_x / np.sum(e_x, axis=1, keepdims=True)

                current_policy = softmax(network_output.policy_logits)
                reusing_node_indices = [-1 for _ in range(num)]

                # Similarity
                if len(policy_softmax_pool) > 1:
                    ### Policy Similarity
                    current_policy_ = np.expand_dims(current_policy, axis=1)
                    grouped_policies_by_num = np.transpose(np.asarray(policy_softmax_pool[1:]), axes=(1, 0, 2)) 
                    policy_sims = np.mean(np.square(grouped_policies_by_num - current_policy_), axis=2)
                    policy_threshold_satisfied = policy_sims < 0.00001
                    
                    ### State Similarity
                    batch_size, num_prev_compares = grouped_policies_by_num.shape[:2]
                    grouped_hidden_states_by_num = np.transpose(np.asarray(hidden_state_pool[1:-1]), axes=(1, 0, 2, 3, 4))
                    state_sims = cosine_similarity(batch_size, num_prev_compares, grouped_hidden_states_by_num, hidden_state_nodes)
                    state_threshold_satisfied = state_sims >= 0.97

                    # Count times that both thresholds were met and get their timestep indices
                    all_threshold_satisfied = np.logical_and(policy_threshold_satisfied, state_threshold_satisfied)
                    all_threshold_satisfied_per_batch = np.sum(all_threshold_satisfied, axis=1) > 0
                    all_threshold_satisfied_timestep_indices = np.argmax(all_threshold_satisfied, axis=1) # All 1's and 0's
                    # Use -1 if not found for updating in back_propagate
                    indices_if_satisfied_else_neg1 = np.where(all_threshold_satisfied_per_batch, all_threshold_satisfied_timestep_indices + 1, np.zeros_like(all_threshold_satisfied_per_batch) - 1)
                    
                    # Update total counter for sim states found
                    # similar_states_found = np.sum(all_threshold_satisfied_per_batch)
                    # total_similar_states_found += similar_states_found
                    # if similar_states_found > 0:
                        # print(f"{all_threshold_satisfied_timestep_indices}, {indices_if_satisfied_else_neg1}")
                        # input(f"Similar states found: {similar_states_found}")

                    # Structure info to pass to cpp code to reuse existing states
                    gathered_policy = grouped_policies_by_num[np.arange(num), all_threshold_satisfied_timestep_indices.reshape(-1), :]
                    new_policy = np.where(np.expand_dims(all_threshold_satisfied_per_batch, axis=1), gathered_policy, current_policy)
                    current_policy = new_policy
                    reusing_node_indices = indices_if_satisfied_else_neg1.reshape(-1).tolist()

                policy_softmax_pool.append(current_policy)
                ############################

                hidden_state_pool.append(hidden_state_nodes)
                # reset 0
                # reset the hidden states in LSTM every horizon steps in search
                # only need to predict the value prefix in a range (eg: s0 -> s5)
                assert horizons > 0
                reset_idx = (np.array(search_lens) % horizons == 0)
                assert len(reset_idx) == num
                reward_hidden_nodes[0][:, reset_idx, :] = 0
                reward_hidden_nodes[1][:, reset_idx, :] = 0
                is_reset_lst = reset_idx.astype(np.int32).tolist()

                reward_hidden_c_pool.append(reward_hidden_nodes[0])
                reward_hidden_h_pool.append(reward_hidden_nodes[1])
                hidden_state_index_x += 1

                # backpropagation along the search path to update the attributes
                tree.batch_back_propagate(reusing_node_indices, hidden_state_index_x, discount,
                                          value_prefix_pool, value_pool, policy_logits_pool,
                                          min_max_stats_lst, results, is_reset_lst)

            # print(f"Total similar states found: {total_similar_states_found}")
