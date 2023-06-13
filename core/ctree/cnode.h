#ifndef CNODE_H
#define CNODE_H

#include "cminimax.h"
#include <math.h>
#include <vector>
#include <stack>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <sys/timeb.h>
#include <sys/time.h>
#include <set>
#include <tuple>

const int DEBUG_MODE = 0;

namespace tree {

    class CNode {
        public:
            int visit_count, to_play, action_num, hidden_state_index_x, hidden_state_index_y, hidden_state_index_z, best_action, is_reset;
            float value_prefix, prior, value_sum;
            std::vector<int> children_index;
            std::vector<CNode>* ptr_node_pool;

            CNode();
            CNode(float prior, int action_num, std::vector<CNode> *ptr_node_pool);
            ~CNode();

            void print();
            void expand(int to_play, int hidden_state_index_x, int hidden_state_index_y, int hidden_state_index_z, float value_prefix, const std::vector<float> &policy_logits);
            void add_exploration_noise(float exploration_fraction, const std::vector<float> &noises);
            float get_mean_q(int isRoot, float parent_q, float discount);
            void print_out();

            int expanded();

            float value();

            std::vector<int> get_trajectory();
            std::vector<int> get_children_distribution();
            std::vector<float> get_children_values();
            CNode* get_child(int action);
    };

    class CRoots{
        public:
            int root_num, action_num, pool_size;
            std::vector<CNode> roots;
            std::vector<std::vector<CNode>> node_pools;

            CRoots();
            CRoots(int root_num, int action_num, int pool_size);
            ~CRoots();

            void prepare(float root_exploration_fraction, const std::vector<std::vector<float>> &noises, const std::vector<float> &value_prefixs, const std::vector<std::vector<float>> &policies);
            void prepare_no_noise(const std::vector<float> &value_prefixs, const std::vector<std::vector<float>> &policies);
            void clear();
            std::vector<std::vector<int>> get_trajectories();
            std::vector<std::vector<int>> get_distributions();
            std::vector<std::vector<float>> get_children_values();
            std::vector<float> get_values();

    };

    class CSearchResults{
        public:
            int num, searches;
            float PUCT_average_score, num_PUCT_scores;
            std::vector<int> num_searched;
            std::vector<std::vector<int>> hidden_state_index_x_lst, hidden_state_index_y_lst, hidden_state_index_z_lst, last_actions, search_lens;
            std::vector<std::vector<CNode*>> nodes;
            std::vector<std::vector<std::vector<CNode*>>> search_paths;

            void print();
            CSearchResults();
            CSearchResults(int num, int searches);
            ~CSearchResults();

    };


    //*********************************************************
    void update_tree_q(CNode* root, tools::CMinMaxStats &min_max_stats, float discount);
    void cback_propagate(std::vector<CNode*> &search_path, tools::CMinMaxStats &min_max_stats, int to_play, float value, float discount);
    void cbatch_back_propagate(int hidden_state_index_x, float discount, const std::vector<std::vector<float>> &value_prefixs, const std::vector<std::vector<float>> &values, const std::vector<std::vector<std::vector<float>>> &policies, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<std::vector<int>> is_reset_lst);
    std::tuple<int, int> cselect_child(CSearchResults* results, CNode* root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount, float mean_q);
    float cucb_score(CNode *child, tools::CMinMaxStats &min_max_stats, float parent_mean_q, int is_reset, float total_children_visit_counts, float parent_value_prefix, float pb_c_base, float pb_c_init, float discount);
    float cbatch_traverse(CRoots *roots, int pb_c_base, float pb_c_init, float discount, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results);
}

#endif