#include <iostream>
#include "parameters.h"
#include "myproject.h"

#include "../../../hls4ml/templates/vivado/nnet_utils/nnet_activation.h"
#include "../../../hls4ml/templates/vivado/nnet_utils/nnet_dense_large.h"
#include "../../../hls4ml/templates/vivado/nnet_utils/nnet_common.h"
#include "../../../hls4ml/templates/vivado/nnet_utils/nnet_helpers.h"
#include "../../../hls4ml/templates/vivado/nnet_utils/nnet_graph.h"

//insert weights from training
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

/* Weight Matrices Dimensions
Encoder (3,128)
Core (256,128) Then concatenate first encoded features for 128 --> 256
Decoder (256,128)
Output (128,1)


//parameters
#define N_NODES 2714
#define N_FEATURES 3
#define N_EDGES 24758
#define E_FEATURES 4
#define LATENT 128
*/

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
#pragma HLS ARRAY_RESHAPE variable=edge_index complete dim=0
#pragma HLS ARRAY_RESHAPE variable=e complete dim=0
#pragma HLS INTERFACE ap_vld port=N,E,edge_index,e
#pragma HLS DATAFLOW

  const_size_in	= N_NODES_MAX*N_FEATURES+N_EDGES_MAX*E_FEATURES+N_EDGES_MAX*2;
  const_size_out = N_EDGES_MAX*1;

#ifndef __SYNTHESIS__
  static bool loaded_weights = false;
 if (!loaded_weights) {
   //hls-fpga-machine-learning insert load weights                                                                           
   nnet::load_weights_from_txt<model_default_t, N_FEATURES*LATENT>(encoder_node_w0, "encoder_node_w0.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT>(encoder_node_b0, "encoder_node_b0.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT*LATENT>(encoder_node_w1, "encoder_node_w1.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT>(encoder_node_b1, "encoder_node_b1.txt");
   nnet::load_weights_from_txt<model_default_t, E_FEATURES*LATENT>(encoder_edge_w0, "encoder_edge_w0.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT>(encoder_edge_b0, "encoder_edge_b0.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT*LATENT>(encoder_edge_w1, "encoder_edge_w1.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT>(encoder_edge_b1, "encoder_edge_b1.txt");
   nnet::load_weights_from_txt<model_default_t, 3*LATENT*LATENT>(core_edge_w0, "core_edge_w0.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT>(core_edge_b0, "core_edge_b0.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT*LATENT>(core_edge_w1, "core_edge_w1.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT>(core_edge_b1, "core_edge_b1.txt");
   nnet::load_weights_from_txt<model_default_t, 1>(core_edge_w2, "core_edge_w2.txt");
   nnet::load_weights_from_txt<model_default_t, 1>(core_edge_b2, "core_edge_b2.txt");
   nnet::load_weights_from_txt<model_default_t, 1>(core_edge_w3, "core_edge_w3.txt");
   nnet::load_weights_from_txt<model_default_t, 1>(core_edge_b3, "core_edge_b3.txt");
   nnet::load_weights_from_txt<model_default_t, 2*LATENT*LATENT>(core_node_w0, "core_node_w0.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT>(core_node_b0, "core_node_b0.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT*LATENT>(core_node_w1, "core_node_w1.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT>(core_node_b1, "core_node_b1.txt");
   nnet::load_weights_from_txt<model_default_t, 1>(core_node_w2, "core_node_w2.txt");
   nnet::load_weights_from_txt<model_default_t, 1>(core_node_b2, "core_node_b2.txt");
   nnet::load_weights_from_txt<model_default_t, 1>(core_node_w3, "core_node_w3.txt");
   nnet::load_weights_from_txt<model_default_t, 1>(core_node_b3, "core_node_b3.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT*LATENT>(decoder_edge_w0, "decoder_edge_w0.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT>(decoder_edge_b0, "decoder_edge_b0.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT*LATENT>(decoder_edge_w1, "decoder_edge_w1.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT>(decoder_edge_b1, "decoder_edge_b1.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT*LATENT>(decoder_edge_w2, "decoder_edge_w2.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT>(decoder_edge_b2, "decoder_edge_b2.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT*1>(decoder_edge_w3, "decoder_edge_w3.txt");
   nnet::load_weights_from_txt<model_default_t, 1>(decoder_edge_b3, "decoder_edge_b3.txt");

   loaded_weights = true;
 }
#endif

  //encode nodes features
  input_t Rn[N_NODES_MAX][LATENT];
  #pragma HLS ARRAY_PARTITION variable=Rn complete dim=0
  nnet::graph_independent<input_t, input_t, graph_config1>(N, Rn, encoder_node_w0, encoder_node_b0, encoder_node_w1, encoder_node_b1);
  /*
  for(int i = 0; i < N_NODES; i++){
    std::cout << "Rn[" << i << "] = [";
    for(int j = 0; j < latent_dim; j++){
      std::cout << Rn[i][j] << ", ";
    }
    std::cout << "]" << std::endl;
  }
  */
  //encode edge features
  input_t Re[N_EDGES_MAX][LATENT];
  #pragma HLS ARRAY_PARTITION variable=Re complete dim=0
  nnet::graph_independent<input_t, input_t, graph_config2>(E, Re, encoder_edge_w0, encoder_edge_b0, encoder_edge_w1, encoder_edge_b1);
  /*
  for(int i = 0; i < N_EDGES; i++){
    std::cout << "Re[" << i << "] = [";
    for(int j = 0; j < latent_dim; j++){
      std::cout << Re[i][j] << ", ";
    }
    std::cout << "]" << std::endl;
  }
  */
  //core networks
  input_t L[N_EDGES_MAX][LATENT];
  input_t Q[N_NODES_MAX][LATENT];
  input_t P[N_NODES_MAX][LATENT];
  #pragma HLS ARRAY_PARTITION variable=L complete dim=0
  #pragma HLS ARRAY_PARTITION variable=Q complete dim=0
  #pragma HLS ARRAY_PARTITION variable=P complete dim=0
  
  for(int i = 0; i < N_ITERS; i++){

    //core edge updates
    nnet::IN_edge_module<input_t, index_t, input_t, graph_config3>(Re, Rn, edge_index, L, Q, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
    //core node updates
    nnet::IN_node_module<input_t, input_t, graph_config4>(Rn, Q, P, core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2, core_node_w3, core_node_b3);

  }

  //decode edge features
  input_t Ro[N_EDGES_MAX][LATENT];
  #pragma HLS ARRAY_PARTITION variable=Ro complete dim=0
  nnet::graph_independent<input_t, input_t, graph_config5>(L, Ro, decoder_edge_w0, decoder_edge_b0, decoder_edge_w1, decoder_edge_b1);

  input_t e_logits[N_EDGES_MAX][1];
  #pragma HLS ARRAY_PARTITION variable=e_logits complete dim=0
  nnet::graph_independent<input_t, input_t, graph_config6>(Ro, e_logits, decoder_edge_w2, decoder_edge_b2, decoder_edge_w3, decoder_edge_b3);

  nnet::sigmoid_batch<input_t, input_t, sigmoid_config1>(e_logits, e);
}
