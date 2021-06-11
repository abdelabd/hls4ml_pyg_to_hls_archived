#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_helpers.h"
//hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_resource.h"
#include "nnet_utils/nnet_graph.h"
#include "nnet_utils/nnet_merge.h"

//hls-fpga-machine-learning insert weights
#include "weights/core_edge_w0.h"
#include "weights/core_edge_b0.h"
#include "weights/core_edge_w1.h"
#include "weights/core_edge_b1.h"
#include "weights/core_edge_w2.h"
#include "weights/core_edge_b2.h"
#include "weights/core_edge_w3.h"
#include "weights/core_edge_b3.h"
#include "weights/core_node_w0.h"
#include "weights/core_node_b0.h"
#include "weights/core_node_w1.h"
#include "weights/core_node_b1.h"
#include "weights/core_node_w2.h"
#include "weights/core_node_b2.h"

#include "defines.h"
//hls-fpga-machine-learning insert layer-config

struct graph_config1 : nnet::graph_config {
  static const unsigned n_edge = N_EDGES_MAX;
  static const unsigned n_node = N_NODES_MAX;
  static const unsigned n_hidden = LATENT_EDGE;
  static const unsigned e_features = E_FEATURES;
  static const unsigned n_features = N_FEATURES;
  static const unsigned n_out = E_FEATURES;
  static const bool io_stream = true;
  static const bool activate_final = false;
  static const unsigned reuse_factor = REUSE_GRAPH;
  static const unsigned n_layers = 3;
  struct merge_config1 : nnet::merge_config {
    static const unsigned n_elem1_0 = E_FEATURES;
    static const unsigned n_elem2_0 = N_FEATURES;
  };
  struct merge_config2 : nnet::merge_config {
    static const unsigned n_elem1_0 = E_FEATURES+N_FEATURES;
    static const unsigned n_elem2_0 = N_FEATURES;
  };
  struct dense_config1 : nnet::dense_config {
    static const unsigned n_in = e_features + 2*n_features;
    static const unsigned n_out = n_hidden;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config1 : nnet::activ_config {
    static const unsigned n_in = n_hidden;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
  };
  struct dense_config2 : nnet::dense_config {
    static const unsigned n_in = n_hidden;
    static const unsigned n_out = n_hidden;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config2 : nnet::activ_config {
    static const unsigned n_in = n_hidden;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
  };
  struct dense_config3 : nnet::dense_config {
    static const unsigned n_in = n_hidden;
    static const unsigned n_out = e_features;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config3 : nnet::activ_config {};
  struct dense_config4 : nnet::dense_config {
    static const unsigned n_in = 1;
    static const unsigned n_out = 1;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config4 : nnet::activ_config {};
};

struct graph_config2 : nnet::graph_config {
  static const unsigned n_edge = N_EDGES_MAX;
  static const unsigned n_node = N_NODES_MAX;
  static const unsigned n_hidden = LATENT_NODE;
  static const unsigned e_features = E_FEATURES;
  static const unsigned n_features = N_FEATURES;
  static const unsigned n_out = N_FEATURES;
  static const bool activate_final = false;
  static const unsigned reuse_factor = REUSE_GRAPH;
  static const unsigned n_layers = 3;
  struct merge_config1 : nnet::merge_config {
    static const unsigned n_elem1_0 = E_FEATURES;
    static const unsigned n_elem2_0 = N_FEATURES;
  };
  struct dense_config1 : nnet::dense_config {
    static const unsigned n_in = e_features + n_features;
    static const unsigned n_out = n_hidden;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config1 : nnet::activ_config {
    static const unsigned n_in = n_hidden;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
  };
  struct dense_config2 : nnet::dense_config {
    static const unsigned n_in = n_hidden;
    static const unsigned n_out = n_hidden;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config_2 : nnet::activ_config {
    static const unsigned n_in = n_hidden;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;    
  };
  struct dense_config3 : nnet::dense_config {
    static const unsigned n_in = n_hidden;
    static const unsigned n_out = n_out; //n_features;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config3 : nnet::activ_config {};
  struct dense_config4 : nnet::dense_config {
    static const unsigned n_in = 1;
    static const unsigned n_out = 1;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config4 : nnet::activ_config {};
};

struct graph_config3 : nnet::graph_config {
  static const unsigned n_edge = N_EDGES_MAX;
  static const unsigned n_node = N_NODES_MAX;
  static const unsigned n_hidden = LATENT_EDGE;
  static const unsigned e_features = E_FEATURES;
  static const unsigned n_features = N_FEATURES;
  static const unsigned n_out = 1;
  static const bool io_stream = false;
  static const bool activate_final = true;
  static const unsigned reuse_factor = REUSE_GRAPH;
  static const unsigned n_layers = 3;
  struct merge_config1 : nnet::merge_config {
    static const unsigned n_elem1_0 = E_FEATURES;
    static const unsigned n_elem2_0 = N_FEATURES;
  };
  struct merge_config2 : nnet::merge_config {
    static const unsigned n_elem1_0 = E_FEATURES+N_FEATURES;
    static const unsigned n_elem2_0 = N_FEATURES;
  };
  struct dense_config1 : nnet::dense_config {
    static const unsigned n_in = e_features + 2*n_features;
    static const unsigned n_out = n_hidden;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config1 : nnet::activ_config {
    static const unsigned n_in = n_hidden;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
  };
  struct dense_config2 : nnet::dense_config {
    static const unsigned n_in = n_hidden;
    static const unsigned n_out = n_hidden;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config2 : nnet::activ_config {
    static const unsigned n_in = n_hidden;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
  };
  struct dense_config3 : nnet::dense_config {
    static const unsigned n_in = n_hidden;
    static const unsigned n_out = n_out; //1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config3 : nnet::activ_config {};
  struct dense_config4 : nnet::dense_config {
    static const unsigned n_in = 1;
    static const unsigned n_out = 1;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
};

struct sigmoid_config1 : nnet::activ_config {
  static const unsigned n_batch = N_EDGES_MAX;
  static const unsigned n_in = 1;
  static const unsigned table_size = 1024;
  static const unsigned io_type = nnet::io_parallel;
};

#endif 
