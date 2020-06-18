#include <iostream>
#include "parameters.h"
#include "myproject.h"

#include "../../../hls4ml/templates/vivado/nnet_utils/nnet_activation.h"
#include "../../../hls4ml/templates/vivado/nnet_utils/nnet_dense_large.h"
#include "../../../hls4ml/templates/vivado/nnet_utils/nnet_common.h"
#include "../../../hls4ml/templates/vivado/nnet_utils/nnet_helpers.h"
#include "../../../hls4ml/templates/vivado/nnet_utils/nnet_graph.h"

//insert weights from training
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

void myproject(
	       input_t      N[N_NODES_MAX][N_FEATURES],
	       input_t      E[N_EDGES_MAX][E_FEATURES],
               index_t      edge_index[N_EDGES_MAX][2],
	       result_t     e[N_EDGES_MAX][1],
	       unsigned short &const_size_in,
	       unsigned short &const_size_out)
{

  //hls-fpga-machine-learning insert IO
#pragma HLS ARRAY_RESHAPE variable=N complete dim=0
#pragma HLS ARRAY_RESHAPE variable=E complete dim=0
#pragma HLS ARRAY_RESHAPE variable=receivers complete dim=0
#pragma HLS ARRAY_RESHAPE variable=senders complete dim=0
#pragma HLS ARRAY_RESHAPE variable=e complete dim=0
#pragma HLS INTERFACE ap_vld port=N,E,receivers,senders,e
//#pragma HLS DATAFLOW

  const_size_in	= N_NODES_MAX*N_FEATURES+N_EDGES_MAX*E_FEATURES+N_EDGES_MAX*2;
  const_size_out = N_EDGES_MAX*1;

#ifndef __SYNTHESIS__
  static bool loaded_weights = false;
 if (!loaded_weights) {
   //hls-fpga-machine-learning insert load weights                                                                           
   nnet::load_weights_from_txt<model_default_t, E_FEATURES*LATENT_EDGE + 2*N_FEATURES*LATENT_EDGE>(core_edge_w0, "core_edge_w0.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT_EDGE>(core_edge_b0, "core_edge_b0.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT_EDGE*LATENT_EDGE>(core_edge_w1, "core_edge_w1.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT_EDGE>(core_edge_b1, "core_edge_b1.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT_EDGE*E_FEATURES>(core_edge_w2, "core_edge_w2.txt");
   nnet::load_weights_from_txt<model_default_t, E_FEATURES>(core_edge_b2, "core_edge_b2.txt");
   nnet::load_weights_from_txt<model_default_t, 1>(core_edge_w3, "core_edge_w3.txt");
   nnet::load_weights_from_txt<model_default_t, 1>(core_edge_b3, "core_edge_b3.txt");
   nnet::load_weights_from_txt<model_default_t, E_FEATURES*LATENT_NODE + N_FEATURES*LATENT_NODE>(core_node_w0, "core_node_w0.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT_NODE>(core_node_b0, "core_node_b0.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT_NODE*LATENT_NODE>(core_node_w1, "core_node_w1.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT_NODE>(core_node_b1, "core_node_b1.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT_NODE*N_FEATURES>(core_node_w2, "core_node_w2.txt");
   nnet::load_weights_from_txt<model_default_t, N_FEATURES>(core_node_b2, "core_node_b2.txt");
   nnet::load_weights_from_txt<model_default_t, 1>(core_node_w3, "core_node_w3.txt");
   nnet::load_weights_from_txt<model_default_t, 1>(core_node_b3, "core_node_b3.txt");

   loaded_weights = true;
 }
#endif

  //interaction network
  input_t edge_update[N_EDGES_MAX][E_FEATURES];
  input_t aggregation[N_NODES_MAX][E_FEATURES];
  input_t node_update[N_NODES_MAX][N_FEATURES];
  #pragma HLS ARRAY_PARTITION variable=effects complete dim=0
  #pragma HLS ARRAY_PARTITION variable=aggregation complete dim=0
  #pragma HLS ARRAY_PARTITION variable=influence complete dim=0

  input_t e_logits[N_EDGES_MAX][1];
  #pragma HLS ARRAY_PARTITION variable=e_logits complete dim=0
  input_t q[N_NODES_MAX][E_FEATURES];
  #pragma HLS ARRAY_PARTITION variable=q complete dim=0
  DO_PRAGMA(HLS PIPELINE II=REUSE_GRAPH)

  //edge block
  nnet::IN_edge_module<input_t, index_t, input_t, graph_config1>(E, N, edge_index, edge_update, aggregation, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
  //node block
nnet::IN_node_module<input_t, input_t, graph_config2>(N, aggregation, node_update, core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2, core_node_w3, core_node_b3);
  //edge block
  nnet::IN_edge_module<input_t, index_t, input_t, graph_config3>(edge_update, node_update, edge_index, e_logits, q, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);

  nnet::sigmoid_batch<input_t, result_t, sigmoid_config1>(e_logits, e);
}
