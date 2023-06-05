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

            total_similar_states_found = 0

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
                # input(current_policy)
                reusing_node_indices = [-1 for _ in range(num)]

                # # Policy Similarity
                # if len(policy_softmax_pool) > 1:
                #     grouped_policies_by_num = np.transpose(np.asarray(policy_softmax_pool[1:]), axes=(1, 0, 2))
                #     # input(grouped_policies_by_num)
                #     # print(grouped_policies_by_num.shape)
                #     current_policy_ = np.expand_dims(current_policy, axis=1)
                #     # input(current_policy_.shape)

                #     MSE = np.mean(np.square(grouped_policies_by_num - current_policy_), axis=2)
                #     # input(MSE.shape)
                #     # print(f"MSE: {MSE}")
                #     min_MSE = np.min(MSE, axis=1)
                #     # print(f"min_MSE: {min_MSE}")
                #     min_under_thr = min_MSE < 0.00001
                #     # if min_under_thr.any():
                #     #     print("threshold reached!")
                #     # print(f"min_under_thr: {min_under_thr}")
                #     argmin_MSE = np.argmin(MSE, axis=1)
                #     # print(f"argmin_MSE: {argmin_MSE}")
                #     # Policies most similar to current policy
                #     gathered_policy = grouped_policies_by_num[np.arange(num), argmin_MSE, :]
                #     # # print(f"gathered_policy: {gathered_policy}")
                #     # # print(f"current_policy: {current_policy}")
                #     x = np.expand_dims(min_under_thr, axis=1)
                #     input(f"min_under_thr.shape{x.shape}")
                #     new_policy = np.where(np.expand_dims(min_under_thr, axis=1), gathered_policy, current_policy)
                #     # print(f"new_policy: {new_policy}")
                #     input(f"new_policy shape:{new_policy.shape}")
                #     current_policy = new_policy
                #     reusing_node_indices = (min_under_thr * (argmin_MSE + 2) - 1).tolist()
                # #     input(f"reusing_node_indices: {reusing_node_indices}")
                # # else:
                # #     input(f"hidden_state_pool[0].shape: {hidden_state_pool[0].shape}")

                # policy_softmax_pool.append(current_policy)
                # # input(policy_softmax_pool)
                # ############################

                # Similarity
                if len(policy_softmax_pool) > 1:
                    grouped_policies_by_num = np.transpose(np.asarray(policy_softmax_pool[1:]), axes=(1, 0, 2))
                    current_policy_ = np.expand_dims(current_policy, axis=1)

                    ### Policy Similarity
                    MSE = np.mean(np.square(grouped_policies_by_num - current_policy_), axis=2)
                    min_MSE = np.min(MSE, axis=1)
                    min_under_thr = min_MSE < 0.00001
                    argmin_MSE = np.argmin(MSE, axis=1)
                    gathered_policy = grouped_policies_by_num[np.arange(num), argmin_MSE, :]
                    
                    ### State Similarity
                    batch_size = grouped_policies_by_num.shape[0]
                    num_prev_compares = grouped_policies_by_num.shape[1]
                    cos_sims = np.zeros((batch_size, num_prev_compares, 1))
                    grouped_hidden_states_by_num = np.transpose(np.asarray(hidden_state_pool[1:-1]), axes=(1, 0, 2, 3, 4))
                    cos_sims = cosine_similarity(batch_size, num_prev_compares, grouped_hidden_states_by_num, hidden_state_nodes)
                    max_cos_sim = np.max(cos_sims, axis=1)
                    max_over_thr = max_cos_sim >= 0.995
                    argmax_cos_sim = np.argmax(cos_sims, axis=1)

                    # Count times that both thresholds were met
                    sim_threshold_met = np.logical_and(min_under_thr, max_over_thr)
                    # indices where policy and state agreed on most similar state
                    policy_and_state_most_sim = argmax_cos_sim == argmin_MSE
                    # Same indices
                    sim_indices = np.logical_and(policy_and_state_most_sim.reshape(-1), sim_threshold_met.reshape(-1))
                    total_similar_states_found += np.sum(sim_indices)

                    num_sim_states = np.sum(max_over_thr)
                    num_sim_policies = np.sum(min_under_thr)
                    if np.sum(sim_threshold_met) > 0:
                        print(f"# Times policy sim: {num_sim_policies}")
                        print(f"# Times states sim: {num_sim_states}")
                        print(f"# Times policy and states sim: {np.sum(sim_threshold_met)}")
                        print(f"# Times policy and state most sim at action X: {np.sum(policy_and_state_most_sim)}")
                        input(f"# Times policy and states sim @ same spot: {np.sum(sim_indices)}")

                    # Structure info to pass to cpp code to reuse existing states
                    new_policy = np.where(np.expand_dims(sim_indices, axis=1), gathered_policy, current_policy)
                    current_policy = new_policy
                    reusing_node_indices = (sim_indices * (argmin_MSE + 2) - 1).tolist()

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

            print(f"Total similar states found: {total_similar_states_found}")
