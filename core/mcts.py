import torch

import numpy as np
import core.ctree.cytree as tree

from torch.cuda.amp import autocast as autocast

import time

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

        # Number of searches per simulation
        searches = 2

        before_search = time.perf_counter()

        with torch.no_grad():
            model.eval()

            # preparation
            num = roots.num
            device = self.config.device
            pb_c_base, pb_c_init, discount = self.config.pb_c_base, self.config.pb_c_init, self.config.discount
            # the data storage of hidden states: storing the states of all the tree nodes
            prev_hidden_state_shape = list(hidden_state_roots.shape)
            hidden_state_shape = [searches] + prev_hidden_state_shape
            hidden_state_pool = [np.vstack([hidden_state_roots] * searches).reshape(hidden_state_shape)]
            # print(hidden_state_pool[0].shape)
            # input()
            # 1 x batch x 64
            # the data storage of value prefix hidden states in LSTM
            reward_hidden_c_pool = [np.vstack([reward_hidden_roots[0]] * searches).reshape(1, searches, num, -1)]
            reward_hidden_h_pool = [np.vstack([reward_hidden_roots[1]] * searches).reshape(1, searches, num, -1)]
            # the index of each layer in the tree
            hidden_state_index_x = 0
            # minimax value storage
            min_max_stats_lst = tree.MinMaxStatsList(num, searches)
            min_max_stats_lst.set_delta(self.config.value_delta_max)
            horizons = self.config.lstm_horizon_len

            # # Moved outside so we can keep track of all nodes across simulations
            # # prepare a result wrapper to transport results between python and c++ parts
            # results = tree.ResultsWrapper(num)

            for index_simulation in range(self.config.num_simulations):
                hidden_states = []
                hidden_states_c_reward = []
                hidden_states_h_reward = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = tree.ResultsWrapper(num, searches)
                # traverse to select actions for each root
                # hidden_state_index_x_lst: the first index of leaf node states in hidden_state_pool
                # hidden_state_index_y_lst: the second index of leaf node states in hidden_state_pool
                # the hidden state of the leaf node is hidden_state_pool[x, y]; value prefix states are the same
                
                before_batch_traverse = time.perf_counter()
                
                hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions = tree.batch_traverse(roots, pb_c_base, pb_c_init, discount, min_max_stats_lst, results)
                
                after_batch_traverse = time.perf_counter()

                # input('finished batch_traverse')

                # print(f'batch_traverse time: {after_batch_traverse - before_batch_traverse} seconds')

                # obtain the search horizon for leaf nodes
                search_lens = results.get_search_len()

                before_model_inf = time.perf_counter()

                # obtain the states for leaf nodes
                for i in range(searches):
                    for ix, iy in zip(hidden_state_index_x_lst[i], hidden_state_index_y_lst[i]):
                        hidden_states.append(hidden_state_pool[ix][i][iy])
                        hidden_states_c_reward.append(reward_hidden_c_pool[ix][0][i][iy])
                        hidden_states_h_reward.append(reward_hidden_h_pool[ix][0][i][iy])

                hidden_states = torch.from_numpy(np.asarray(hidden_states)).to(device).float()
                hidden_states_c_reward = torch.from_numpy(np.asarray(hidden_states_c_reward)).to(device).unsqueeze(0)
                hidden_states_h_reward = torch.from_numpy(np.asarray(hidden_states_h_reward)).to(device).unsqueeze(0)

                last_actions = torch.from_numpy(np.asarray(last_actions)).to(device).view(-1).unsqueeze(1).long()

                # print(f'hidden_states shape: {hidden_states.shape}, last_actions shape: {last_actions.shape}')
                # print(f'hidden_states_c_reward shape: {hidden_states_c_reward.shape}, hidden_states_h_reward shape: {hidden_states_h_reward.shape}')
                # input('printing hidden_states and last_actions in mcts.py')

                # evaluation for leaf nodes
                if self.config.amp_type == 'torch_amp':
                    with autocast():
                        network_output = model.recurrent_inference(hidden_states, (hidden_states_c_reward, hidden_states_h_reward), last_actions)
                else:
                    network_output = model.recurrent_inference(hidden_states, (hidden_states_c_reward, hidden_states_h_reward), last_actions)

                after_model_inf = time.perf_counter()
                # print(f'model inference time: {after_model_inf - before_model_inf} seconds')

                # print(hidden_state_pool[0].shape)
                # print(hidden_state_index_y_lst, hidden_state_index_x_lst)
                # print(search_lens)
                # results.print_nodes()
                # print('\n', network_output.value)
                # print(network_output.value_prefix)
                # print(network_output.policy_logits)
                # x = network_output.policy_logits
                # softmax_policy = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
                # V = softmax_policy

                # kl = np.dot(V, np.log(V).T)
                # right = kl + kl.T
                # left = np.tile(np.diag(kl),(kl.shape[0],1))
                # left = left + left.T
                # L = left - right

                # np.set_printoptions(precision=4)
                # print(f'Softmax policy: {softmax_policy}')
                # print(f'pairwise KL divergence:\n {L}')
                # # print(network_output.hidden_state.shape)
                # # print(network_output.reward_hidden[0].shape, network_output.reward_hidden[1].shape, len(network_output.reward_hidden))
                # print('KL divergences:')
                # import torch.nn.functional as F
                # for i in range(num-1):
                #     print(softmax_policy[i], softmax_policy[i+1])
                #     KL = F.kl_div(softmax_policy[i].log(), softmax_policy[i+1])
                #     print(float(KL))
                # input(f'printing roots and network_output in mcts.py')

                # print(network_output.value_prefix.shape, network_output.value.shape, network_output.policy_logits.shape)

                hidden_state_nodes = network_output.hidden_state.reshape(hidden_state_shape)
                value_prefix_pool = network_output.value_prefix.reshape(searches, -1).tolist()
                value_pool = network_output.value.reshape(searches, -1).tolist()
                policy_logits_pool = network_output.policy_logits.reshape(searches, num, -1).tolist()
                reward_hidden_nodes = network_output.reward_hidden

                # print(value_prefix_pool, value_pool, policy_logits_pool)
                # print(len(value_prefix_pool), len(value_pool), len(policy_logits_pool))
                # input('printing value_prefix_pool, value_pool, policy_logits_pool in mcts.py')
                # print(search_lens)

                # print(f'hidden_state_pool[0].shape: {hidden_state_pool[0].shape}')
                # print(f'len(hidden_state_pool): {len(hidden_state_pool)}')
                # print(reward_hidden_nodes[0].shape)
                # print(f'hidden_state_nodes.shape: {hidden_state_nodes.shape}')
                # input()

                hidden_state_pool.append(hidden_state_nodes)
                # reset 0
                # reset the hidden states in LSTM every horizon steps in search
                # only need to predict the value prefix in a range (eg: s0 -> s5)
                assert horizons > 0
                reset_idx = (np.array(search_lens) % horizons == 0)
                assert reset_idx.shape[0] == searches and reset_idx.shape[1] == num

                # print(reward_hidden_c_pool[0].shape, reward_hidden_h_pool[0].shape)
                # print(len(reward_hidden_c_pool), len(reward_hidden_h_pool))

                reward_hidden_nodes_0 = reward_hidden_nodes[0].reshape(1, searches, num, -1)
                reward_hidden_nodes_1 = reward_hidden_nodes[1].reshape(1, searches, num, -1)

                reward_hidden_nodes_0[:, reset_idx, :] = 0
                reward_hidden_nodes_1[:, reset_idx, :] = 0
                is_reset_lst = reset_idx.astype(np.int32).tolist()

                reward_hidden_c_pool.append(reward_hidden_nodes_0)
                reward_hidden_h_pool.append(reward_hidden_nodes_1)
                hidden_state_index_x += 1

                # print(is_reset_lst)
                input('before batch_back_propagate in mcts.py')

                # backpropagation along the search path to update the attributes
                tree.batch_back_propagate(hidden_state_index_x, discount,
                                          value_prefix_pool, value_pool, policy_logits_pool,
                                          min_max_stats_lst, results, is_reset_lst)
                
        after_search = time.perf_counter()
        print(f'search time: {after_search - before_search} seconds')
        input('after search input')
