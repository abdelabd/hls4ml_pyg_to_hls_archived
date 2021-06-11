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
#include "weights/encoder_node_w0.h"
#include "weights/encoder_node_b0.h"
#include "weights/encoder_node_w1.h"
#include "weights/encoder_node_b1.h"
#include "weights/encoder_edge_w0.h"
#include "weights/encoder_edge_b0.h"
#include "weights/encoder_edge_w1.h"
#include "weights/encoder_edge_b1.h"
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
#include "weights/core_node_w3.h"
#include "weights/core_node_b3.h"
#include "weights/decoder_edge_w0.h"
#include "weights/decoder_edge_b0.h"
#include "weights/decoder_edge_w1.h"
#include "weights/decoder_edge_b1.h"
#include "weights/decoder_edge_w2.h"
#include "weights/decoder_edge_b2.h"
#include "weights/decoder_edge_w3.h"
#include "weights/decoder_edge_b3.h"

#include "defines.h"
//hls-fpga-machine-learning insert layer-config

struct graph_config1 : nnet::graph_config {
  static const bool io_stream = true;
  static const bool activate_final = true;
  static const unsigned reuse_factor = REUSE_GRAPH;
  typedef bias_default_t bias_t;
  typedef weight_default_t weight_t;

  struct dense_config1 : nnet::dense_config {
    static const unsigned n_batch = N_NODES_MAX;
    static const unsigned n_in = N_FEATURES;
    static const unsigned n_out = LATENT;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config1 : nnet::activ_config {
    static const unsigned n_batch = N_NODES_MAX;
    static const unsigned n_in = LATENT;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
  };
  struct dense_config2 : nnet::dense_config {
    static const unsigned n_batch = N_NODES_MAX;
    static const unsigned n_in = LATENT;
    static const unsigned n_out = LATENT;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config2 : nnet::activ_config {
    static const unsigned n_batch = N_NODES_MAX;
    static const unsigned n_in = LATENT;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
  };
  struct dense_config3 : nnet::dense_config {
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config3 : nnet::activ_config {};
  struct dense_config4 : nnet::dense_config {
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
};

struct graph_config2 : nnet::graph_config {
  static const bool io_stream = true;
  static const bool activate_final = true;
  static const unsigned reuse_factor = REUSE_GRAPH;
  typedef bias_default_t bias_t;
  typedef weight_default_t weight_t;

  struct dense_config1 : nnet::dense_config {
    static const unsigned n_batch = N_EDGES_MAX;
    static const unsigned n_in = E_FEATURES;
    static const unsigned n_out = LATENT;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config1 : nnet::activ_config {
    static const unsigned n_batch = N_EDGES_MAX;
    static const unsigned n_in = LATENT;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
  };
  struct dense_config2 : nnet::dense_config {
    static const unsigned n_batch = N_EDGES_MAX;
    static const unsigned n_in = LATENT;
    static const unsigned n_out = LATENT;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config2 : nnet::activ_config {
    static const unsigned n_batch = N_EDGES_MAX;
    static const unsigned n_in = LATENT;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
  };
  struct dense_config3 : nnet::dense_config {
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config3 : nnet::activ_config {};
  struct dense_config4 : nnet::dense_config {
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
};

struct graph_config3 : nnet::graph_config {
  static const unsigned n_edge = N_EDGES_MAX;
  static const unsigned n_node = N_NODES_MAX;
  static const unsigned n_features = LATENT;
  static const unsigned e_features = LATENT;
  static const unsigned n_out = LATENT;
  static const bool io_stream = true;
  static const bool activate_final = true;
  static const unsigned reuse_factor = REUSE_GRAPH;
  static const unsigned n_layers = 2;
  typedef bias_default_t bias_t;
  typedef weight_default_t weight_t;
  struct merge_config1 : nnet::merge_config {
    static const unsigned n_elem1_0 = e_features;
    static const unsigned n_elem2_0 = n_features;
  };
  struct merge_config2 : nnet::merge_config {
    static const unsigned n_elem1_0 = e_features+n_features;
    static const unsigned n_elem2_0 = n_features;
  };
  struct dense_config1 : nnet::dense_config {
    static const unsigned n_in = e_features + 2*n_features;
    static const unsigned n_out = LATENT;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config1 : nnet::activ_config {
    static const unsigned n_in = LATENT;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
  };
  struct dense_config2 : nnet::dense_config {
    static const unsigned n_in = LATENT;
    static const unsigned n_out = LATENT;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config2 : nnet::activ_config {
    static const unsigned n_in = LATENT;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
  };
  struct dense_config3 : nnet::dense_config {
    static const unsigned n_in = 1;
    static const unsigned n_out = 1;
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

struct graph_config4 : nnet::graph_config {
  static const unsigned n_edge = N_EDGES_MAX;
  static const unsigned n_node = N_NODES_MAX;
  static const unsigned e_features = LATENT;
  static const unsigned n_features = LATENT;
  static const unsigned n_hidden = LATENT;
  static const unsigned n_out = LATENT;
  static const bool activate_final = true;
  static const unsigned reuse_factor = REUSE_GRAPH;
  static const unsigned n_layers = 2;
  typedef bias_default_t bias_t;
  typedef weight_default_t weight_t;
  struct merge_config1 : nnet::merge_config {
    static const unsigned n_elem1_0 = e_features;
    static const unsigned n_elem2_0 = n_features;
  };
  struct dense_config1 : nnet::dense_config {
    static const unsigned n_in = e_features + n_features;
    static const unsigned n_out = LATENT;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config1 : nnet::activ_config {
    static const unsigned n_in = LATENT;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
  };
  struct dense_config2 : nnet::dense_config {
    static const unsigned n_in = LATENT;
    static const unsigned n_out = LATENT;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config2 : nnet::activ_config {
    static const unsigned n_in = LATENT;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
  };
  struct dense_config3 : nnet::dense_config {
    static const unsigned n_in = 1;
    static const unsigned n_out = 1;
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

struct graph_config5 : nnet::graph_config {
  static const bool io_stream = false;
  static const bool activate_final = true;
  static const unsigned reuse_factor = REUSE_GRAPH;
  typedef bias_default_t bias_t;
  typedef weight_default_t weight_t;

  struct dense_config1 : nnet::dense_config {
    static const unsigned n_batch = N_EDGES_MAX;
    static const unsigned n_in = LATENT;
    static const unsigned n_out = LATENT;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config1 : nnet::activ_config {
    static const unsigned n_batch = N_EDGES_MAX;
    static const unsigned n_in = LATENT;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
  };
  struct dense_config2 : nnet::dense_config {
    static const unsigned n_batch = N_EDGES_MAX;
    static const unsigned n_in = LATENT;
    static const unsigned n_out = LATENT;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config2 : nnet::activ_config {
    static const unsigned n_batch = N_EDGES_MAX;
    static const unsigned n_in = LATENT;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
  };
  struct dense_config3 : nnet::dense_config {
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config3 : nnet::activ_config {};
  struct dense_config4 : nnet::dense_config {
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
};

struct graph_config6 : nnet::graph_config {
  static const bool io_stream = false;
  static const bool activate_final = false;
  static const unsigned reuse_factor = REUSE_GRAPH;
  typedef bias_default_t bias_t;
  typedef weight_default_t weight_t;

  struct dense_config1 : nnet::dense_config {
    static const unsigned n_batch = N_EDGES_MAX;
    static const unsigned n_in = LATENT;
    static const unsigned n_out = LATENT;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config1 : nnet::activ_config {
    static const unsigned n_batch = N_EDGES_MAX;
    static const unsigned n_in = LATENT;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
  };
  struct dense_config2 : nnet::dense_config {
    static const unsigned n_batch = N_EDGES_MAX;
    static const unsigned n_in = LATENT;
    static const unsigned n_out = 1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config2 : nnet::activ_config {
    static const unsigned n_batch = N_EDGES_MAX;
    static const unsigned n_in = 1;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
  };
  struct dense_config3 : nnet::dense_config {
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config3 : nnet::activ_config {};
  struct dense_config4 : nnet::dense_config {
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
  static const unsigned reuse_factor = REUSE_DENSE;
};

#endif 
