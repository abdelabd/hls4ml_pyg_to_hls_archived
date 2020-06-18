//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef NNET_GRAPH_H_
#define NNET_GRAPH_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_dense_large.h"
#include "nnet_activation.h"
#include "nnet_conv.h"
#include "hls_stream.h"
#include <math.h>

namespace nnet {
  
  struct graph_config
  {
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    
    // Layer Sizes
    static const unsigned n_node = 10;
    static const unsigned n_edge = 20;
    static const unsigned n_batch = 20;
    static const unsigned n_in = 7;
    static const unsigned n_hidden = 4;
    static const unsigned n_out = 4;
    
    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned io_stream = false;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
  };

  template<class data_T, class res_T, typename CONFIG_T>
    void dense_mult_1lyr(
			 data_T data[CONFIG_T::dense_config1::n_in],
			 res_T res[CONFIG_T::dense_config1::n_out],
			 typename CONFIG_T::dense_config1::weight_t weights0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			 typename CONFIG_T::dense_config1::bias_t   biases0[CONFIG_T::dense_config1::n_out])
  {
    nnet::dense_large_basic<data_T, res_T, typename CONFIG_T::dense_config1>(data, res, weights0, biases0);
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void dense_mult_2lyr(
			 data_T data[CONFIG_T::dense_config1::n_in],
			 res_T res[CONFIG_T::dense_config2::n_out],
			 typename CONFIG_T::dense_config1::weight_t weights0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			 typename CONFIG_T::dense_config1::bias_t   biases0[CONFIG_T::dense_config1::n_out],
			 typename CONFIG_T::dense_config2::weight_t weights1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
			 typename CONFIG_T::dense_config2::bias_t   biases1[CONFIG_T::dense_config2::n_out])
  {
    data_T data0_logits[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0_logits complete dim=0
    nnet::dense_large_basic<data_T, data_T, typename CONFIG_T::dense_config1>(data, data0_logits, weights0, biases0);
    data_T data0[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config1>(data0_logits, data0);

    nnet::dense_large_basic<data_T, res_T, typename CONFIG_T::dense_config2>(data0, res, weights1, biases1);
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void dense_mult_3lyr(
			 data_T data[CONFIG_T::dense_config1::n_in],
			 res_T res[CONFIG_T::dense_config3::n_out],
			 typename CONFIG_T::dense_config1::weight_t weights0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			 typename CONFIG_T::dense_config1::bias_t   biases0[CONFIG_T::dense_config1::n_out],
			 typename CONFIG_T::dense_config2::weight_t weights1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
			 typename CONFIG_T::dense_config2::bias_t   biases1[CONFIG_T::dense_config2::n_out],
			 typename CONFIG_T::dense_config3::weight_t weights2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
			 typename CONFIG_T::dense_config3::bias_t   biases2[CONFIG_T::dense_config3::n_out])
  {
    data_T data0_logits[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0_logits complete dim=0
    nnet::dense_large_basic<data_T, data_T, typename CONFIG_T::dense_config1>(data, data0_logits, weights0, biases0);
    data_T data0[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config1>(data0_logits, data0);

    data_T data1_logits[CONFIG_T::dense_config2::n_out];
    #pragma HLS ARRAY_PARTITION variable=data1_logits complete dim=0
    nnet::dense_large_basic<data_T, data_T, typename CONFIG_T::dense_config2>(data0, data1_logits, weights1, biases1);
    data_T data1[CONFIG_T::dense_config2::n_out];
    #pragma HLS ARRAY_PARTITION variable=data1 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config2>(data1_logits, data1);

    nnet::dense_large_basic<data_T, res_T, typename CONFIG_T::dense_config3>(data1, res, weights2, biases2);
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void dense_mult_4lyr(
			 data_T data[CONFIG_T::dense_config1::n_in],
			 res_T res[CONFIG_T::dense_config4::n_out],
			 typename CONFIG_T::dense_config1::weight_t weights0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			 typename CONFIG_T::dense_config1::bias_t   biases0[CONFIG_T::dense_config1::n_out],
			 typename CONFIG_T::dense_config2::weight_t weights1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
			 typename CONFIG_T::dense_config2::bias_t   biases1[CONFIG_T::dense_config2::n_out],
			 typename CONFIG_T::dense_config3::weight_t weights2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
			 typename CONFIG_T::dense_config3::bias_t   biases2[CONFIG_T::dense_config3::n_out],
			 typename CONFIG_T::dense_config4::weight_t weights3[CONFIG_T::dense_config4::n_in*CONFIG_T::dense_config4::n_out],
			 typename CONFIG_T::dense_config4::bias_t   biases3[CONFIG_T::dense_config4::n_out])
  {
    data_T data0_logits[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0_logits complete dim=0
    nnet::dense_large_basic<data_T, data_T, typename CONFIG_T::dense_config1>(data, data0_logits, weights0, biases0);
    data_T data0[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config1>(data0_logits, data0);

    data_T data1_logits[CONFIG_T::dense_config2::n_out];
    #pragma HLS ARRAY_PARTITION variable=data1_logits complete dim=0
    nnet::dense_large_basic<data_T, data_T, typename CONFIG_T::dense_config2>(data0, data1_logits, weights1, biases1);
    data_T data1[CONFIG_T::dense_config2::n_out];
    #pragma HLS ARRAY_PARTITION variable=data1 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config2>(data1_logits, data1);

    data_T data2_logits[CONFIG_T::dense_config3::n_out];
    #pragma HLS ARRAY_PARTITION variable=data2_logits complete dim=0
    nnet::dense_large_basic<data_T, data_T, typename CONFIG_T::dense_config3>(data1, data2_logits, weights2, biases2);
    data_T data2[CONFIG_T::dense_config3::n_out];
    #pragma HLS ARRAY_PARTITION variable=data2 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config3>(data2_logits, data2);

    nnet::dense_large_basic<data_T, res_T, typename CONFIG_T::dense_config4>(data2, res, weights3, biases3);
  }

  template<class data_T, class index_T, class res_T, typename CONFIG_T>
    void IN_edge_module(
			data_T    Re[CONFIG_T::n_edge][CONFIG_T::e_features],
			data_T    Rn[CONFIG_T::n_node][CONFIG_T::n_features],
			index_T   edge_index[CONFIG_T::n_edge][2],
			res_T     L[CONFIG_T::n_edge][CONFIG_T::n_out],
			res_T     Q[CONFIG_T::n_node][CONFIG_T::n_out],
			typename CONFIG_T::dense_config1::weight_t  core_edge_w0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config1::bias_t    core_edge_b0[CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config2::weight_t  core_edge_w1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config2::bias_t    core_edge_b1[CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config3::weight_t  core_edge_w2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config3::bias_t    core_edge_b2[CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config4::weight_t  core_edge_w3[CONFIG_T::dense_config4::n_in*CONFIG_T::dense_config4::n_out],
			typename CONFIG_T::dense_config4::bias_t    core_edge_b3[CONFIG_T::dense_config4::n_out])
  {
    if(CONFIG_T::io_stream){
      #pragma HLS STREAM variable=edge_index
    }
    
    for(int i = 0; i < CONFIG_T::n_node; i++){
      for(int j = 0; j < CONFIG_T::n_out; j++){
	Q[i][j] = 0;
      }
    }

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor //*CONFIG_T::n_edge
    IN_edge_loop: for(int i = 0; i < CONFIG_T::n_edge; i++) {
      #pragma HLS UNROLL
      index_T r = edge_index[i][0];
      index_T s = edge_index[i][1];
      data_T l_logits[CONFIG_T::e_features+CONFIG_T::n_features];
      #pragma HLS ARRAY_PARTITION variable=l_logits complete dim=0
      nnet::merge<data_T, CONFIG_T::e_features, CONFIG_T::n_features>(Re[i], Rn[r], l_logits);
      data_T l[CONFIG_T::e_features + 2*CONFIG_T::n_features];
      #pragma HLS ARRAY_PARTITION variable=l complete dim=0
      nnet::merge<data_T, CONFIG_T::e_features + CONFIG_T::n_features, CONFIG_T::n_features>(l_logits, Rn[s], l);
      /*
      data_T L0_logits[CONFIG_T::dense_config1::n_out];
      #pragma HLS ARRAY_PARTITION variable=L0_logits complete dim=0
      nnet::dense_large_basic<data_T, data_T, typename CONFIG_T::dense_config1>(l, L0_logits, core_edge_w0, core_edge_b0);
      data_T L0[CONFIG_T::relu_config1::n_in];
      #pragma HLS ARRAY_PARTITION variable=L0 complete dim=0
      nnet::relu<data_T, data_T, typename CONFIG_T::relu_config1>(L0_logits, L0);

      data_T L_logits[CONFIG_T::dense_config2::n_out];
      #pragma HLS ARRAY_PARTITION variable=L_logits complete dim=0
      nnet::dense_large_basic<data_T, data_T, typename CONFIG_T::dense_config2>(L0, L_logits, core_edge_w1, core_edge_b1);
      nnet::relu<data_T, res_T, typename CONFIG_T::relu_config2>(L_logits, L[i]);
      */
      if(CONFIG_T::activate_final){
	data_T L_logits[CONFIG_T::n_out];
        #pragma HLS ARRAY_PARTITION variable=L_logits complete dim=0
        if(CONFIG_T::n_layers == 1){
	  nnet::dense_mult_1lyr<data_T, data_T, CONFIG_T>(l, L_logits, core_edge_w0, core_edge_b0);
	  nnet::relu<data_T, res_T, typename CONFIG_T::relu_config1>(L_logits, L[i]);
        }else if(CONFIG_T::n_layers == 2){
	  nnet::dense_mult_2lyr<data_T, data_T, CONFIG_T>(l, L_logits, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1);
	  nnet::relu<data_T, res_T, typename CONFIG_T::relu_config2>(L_logits, L[i]);
	}else if(CONFIG_T::n_layers == 3){
	  nnet::dense_mult_3lyr<data_T, data_T, CONFIG_T>(l, L_logits, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2);
	  nnet::relu<data_T, res_T, typename CONFIG_T::relu_config3>(L_logits, L[i]);
	}else if(CONFIG_T::n_layers == 4){
	  nnet::dense_mult_4lyr<data_T, data_T, CONFIG_T>(l, L_logits, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
	  nnet::relu<data_T, res_T, typename CONFIG_T::relu_config4>(L_logits, L[i]);
	}
      }else{
        if(CONFIG_T::n_layers == 1){
	  nnet::dense_mult_1lyr<data_T, data_T, CONFIG_T>(l, L[i], core_edge_w0, core_edge_b0);
        }else if(CONFIG_T::n_layers == 2){
	  nnet::dense_mult_2lyr<data_T, data_T, CONFIG_T>(l, L[i], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1);
        }else if(CONFIG_T::n_layers == 3){
	  nnet::dense_mult_3lyr<data_T, data_T, CONFIG_T>(l, L[i], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2);
        }else if(CONFIG_T::n_layers == 4){
	  nnet::dense_mult_4lyr<data_T, res_T, CONFIG_T>(l, L[i], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
        }
      }

      for(int j = 0; j < CONFIG_T::n_out; j++){
        #pragma HLS UNROLL
	Q[r][j] += L[i][j];
      }
    }
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void IN_node_module(
			data_T    Rn[CONFIG_T::n_node][CONFIG_T::n_features],
			data_T    Q[CONFIG_T::n_node][CONFIG_T::e_features],
			res_T     P[CONFIG_T::n_node][CONFIG_T::n_out],
			typename CONFIG_T::dense_config1::weight_t  core_node_w0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config1::bias_t    core_node_b0[CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config2::weight_t  core_node_w1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config2::bias_t    core_node_b1[CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config3::weight_t  core_node_w2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config3::bias_t    core_node_b2[CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config4::weight_t  core_node_w3[CONFIG_T::dense_config4::n_in*CONFIG_T::dense_config4::n_out],
			typename CONFIG_T::dense_config4::bias_t    core_node_b3[CONFIG_T::dense_config4::n_out])
  {
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor //*CONFIG_T::n_node
    IN_node_loop: for(int i = 0; i < CONFIG_T::n_node; i++){
      #pragma HLS UNROLL
      data_T p[CONFIG_T::e_features + CONFIG_T::n_features];
      #pragma HLS ARRAY_PARTITION variable=p complete dim=0
      nnet::merge<data_T, CONFIG_T::e_features, CONFIG_T::n_features>(Q[i], Rn[i], p);
      /*      
      data_T P0_logits[CONFIG_T::dense_config1::n_out];
      #pragma HLS ARRAY_PARTITION variable=P0_logits complete dim=0
      nnet::dense_large_basic<data_T, data_T, typename CONFIG_T::dense_config1>(p, P0_logits, core_node_w0, core_node_b0);
      data_T P0[CONFIG_T::relu_config1::n_in];
      #pragma HLS ARRAY_PARTITION variable=P0 complete dim=0
      nnet::relu<data_T, data_T, typename CONFIG_T::relu_config1>(P0_logits, P0);
      
      data_T P_logits[CONFIG_T::dense_config2::n_out];
      #pragma HLS ARRAY_PARTITION variable=P_logits complete dim=0
      nnet::dense_large_basic<data_T, data_T, typename CONFIG_T::dense_config2>(P0, P_logits, core_node_w1, core_node_b1);
      nnet::relu<data_T, res_T, typename CONFIG_T::relu_config1>(P_logits, P[i]);
      */
      if(CONFIG_T::activate_final){
	data_T P_logits[CONFIG_T::n_out];
	#pragma HLS ARRAY_PARTITION variable=P_logits complete dim=0
	if(CONFIG_T::n_layers == 1){
	  nnet::dense_mult_1lyr<data_T, data_T, CONFIG_T>(p, P_logits, core_node_w0, core_node_b0);
	  nnet::relu<data_T, res_T, typename CONFIG_T::relu_config1>(P_logits, P[i]);
	}else if(CONFIG_T::n_layers == 2){
	  nnet::dense_mult_2lyr<data_T, data_T, CONFIG_T>(p, P_logits, core_node_w0, core_node_b0, core_node_w1, core_node_b1);
	  nnet::relu<data_T, res_T, typename CONFIG_T::relu_config2>(P_logits, P[i]);
	}else if(CONFIG_T::n_layers == 3){
	  nnet::dense_mult_3lyr<data_T, data_T, CONFIG_T>(p, P_logits, core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2);
	  nnet::relu<data_T, res_T, typename CONFIG_T::relu_config3>(P_logits, P[i]);
	}else if(CONFIG_T::n_layers == 4){
	  nnet::dense_mult_4lyr<data_T, data_T, CONFIG_T>(p, P_logits, core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2, core_node_w3, core_node_b3);
	  nnet::relu<data_T, res_T, typename CONFIG_T::relu_config4>(P_logits, P[i]);
	}
      }else{
        if(CONFIG_T::n_layers == 1){
	  nnet::dense_mult_1lyr<data_T, res_T, CONFIG_T>(p, P[i], core_node_w0, core_node_b0);
        }else if(CONFIG_T::n_layers == 2){
	  nnet::dense_mult_2lyr<data_T, res_T, CONFIG_T>(p, P[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1);
        }else if(CONFIG_T::n_layers == 3){
	  nnet::dense_mult_3lyr<data_T, res_T, CONFIG_T>(p, P[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2);
        }else if(CONFIG_T::n_layers == 4){
	  nnet::dense_mult_4lyr<data_T, res_T, CONFIG_T>(p, P[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2, core_node_w3, core_node_b3);
        }
      }
    }
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void graph_independent(
			   data_T    X[CONFIG_T::dense_config1::n_batch][CONFIG_T::dense_config1::n_in],
			   res_T     R[CONFIG_T::dense_config2::n_batch][CONFIG_T::dense_config2::n_out],
			   typename CONFIG_T::dense_config1::weight_t  w0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			   typename CONFIG_T::dense_config1::bias_t    b0[CONFIG_T::dense_config1::n_out],
			   typename CONFIG_T::dense_config2::weight_t  w1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
			   typename CONFIG_T::dense_config2::bias_t    b1[CONFIG_T::dense_config2::n_out])
  {
    if(CONFIG_T::io_stream){
      #pragma HLS STREAM variable=X
    }
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor //*CONFIG_T::dense_config2::n_batch
    data_T R0_logits[CONFIG_T::dense_config1::n_batch][CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=R0_logits complete dim=0
    nnet::dense_batch<data_T, data_T, typename CONFIG_T::dense_config1>(X, R0_logits, w0, b0);
    data_T R0[CONFIG_T::relu_config1::n_batch][CONFIG_T::relu_config1::n_in];
    #pragma HLS ARRAY_PARTITION variable=R0 complete dim=0
    nnet::relu_batch<data_T, data_T, typename CONFIG_T::relu_config1>(R0_logits, R0);

    if(CONFIG_T::activate_final){
        data_T R_logits[CONFIG_T::dense_config2::n_batch][CONFIG_T::dense_config2::n_out];
        #pragma HLS ARRAY_PARTITION variable=R_logits complete dim=0
        nnet::dense_batch<data_T, data_T, typename CONFIG_T::dense_config2>(R0, R_logits, w1, b1);
        nnet::relu_batch<data_T, res_T, typename CONFIG_T::relu_config2>(R_logits, R);
    }else{
      nnet::dense_batch<data_T, data_T, typename CONFIG_T::dense_config2>(R0, R, w1, b1);
    }
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void compute_edge_net_features(
		   data_T    X[CONFIG_T::n_node][CONFIG_T::n_input_dim],
		   ap_uint<1> Ri[CONFIG_T::n_node][CONFIG_T::n_edge],
		   ap_uint<1> Ro[CONFIG_T::n_node][CONFIG_T::n_edge],
		   res_T     B[CONFIG_T::n_edge][2*CONFIG_T::n_input_dim])
  {
    data_T bo[CONFIG_T::n_edge][CONFIG_T::n_input_dim];
    data_T bi[CONFIG_T::n_edge][CONFIG_T::n_input_dim];
    
    if (CONFIG_T::io_type == io_parallel){
      // For parallel inputs:
      //   - completely partition arrays -- target fabric
      //   - if we have an unroll factor, limit number of multipliers
    	#pragma HLS PIPELINE II=CONFIG_T::reuse_factor
		#pragma HLS ARRAY_PARTITION variable=bo complete
		#pragma HLS ARRAY_PARTITION variable=bi complete
    }
    else if (CONFIG_T::io_type == io_serial){
		#pragma HLS DATAFLOW
		#pragma HLS STREAM variable=bo depth=1
		#pragma HLS STREAM variable=bi depth=1
    }

    // Multiply bo = Ro^T X
    // Multiply bi = Ri^T X
    for(int ii = 0; ii < CONFIG_T::n_edge; ii++) {
      if (CONFIG_T::io_type == io_serial){
    	  #pragma HLS PIPELINE
      }
      for(int jj = 0; jj < CONFIG_T::n_input_dim; jj++) {
    	  bi[ii][jj] = 0;
    	  bo[ii][jj] = 0;
    	  for(int kk = 0; kk < CONFIG_T::n_node; kk++) {
    		  bi[ii][jj] += Ri[kk][ii] * X[kk][jj];
    		  bo[ii][jj] += Ro[kk][ii] * X[kk][jj];
    	  }
    	  // Concatenate M = [bo, bi]
    	  B[ii][jj] = (res_T) bo[ii][jj];
    	  B[ii][CONFIG_T::n_input_dim+jj] = (res_T) bi[ii][jj];
      }
    }
  }

  template<class data_T, class res_T, typename CONFIG_T>
  void compute_node_net_features(
		  data_T    X[CONFIG_T::n_node][CONFIG_T::n_input_dim],
		  data_T    e[CONFIG_T::n_edge][1],
		  ap_uint<1> Ri[CONFIG_T::n_node][CONFIG_T::n_edge],
		  ap_uint<1> Ro[CONFIG_T::n_node][CONFIG_T::n_edge],
		  res_T     M[CONFIG_T::n_node][3*CONFIG_T::n_input_dim])
  {
	  data_T bo[CONFIG_T::n_edge][CONFIG_T::n_input_dim];
	  data_T bi[CONFIG_T::n_edge][CONFIG_T::n_input_dim];
	  data_T mo[CONFIG_T::n_node][CONFIG_T::n_input_dim];
	  data_T mi[CONFIG_T::n_node][CONFIG_T::n_input_dim];
	  data_T Rwo[CONFIG_T::n_node][CONFIG_T::n_edge];
	  data_T Rwi[CONFIG_T::n_node][CONFIG_T::n_edge];

	  if (CONFIG_T::io_type == io_parallel){
      // For parallel inputs:
      //   - completely partition arrays -- target fabric
      //   - if we have an unroll factor, limit number of multipliers
		  #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
		  #pragma HLS ARRAY_PARTITION variable=bo complete
		  #pragma HLS ARRAY_PARTITION variable=bi complete
		  #pragma HLS ARRAY_PARTITION variable=mo complete
		  #pragma HLS ARRAY_PARTITION variable=mi complete
		  #pragma HLS ARRAY_PARTITION variable=Rwo complete
		  #pragma HLS ARRAY_PARTITION variable=Rwi complete

		  int multiplier_limit  = ceil(float(2*CONFIG_T::n_node*CONFIG_T::n_edge*CONFIG_T::n_input_dim) / float(CONFIG_T::reuse_factor));
		  #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
	  }
	  else if (CONFIG_T::io_type == io_serial)
	  {
		  #pragma HLS DATAFLOW
		  #pragma HLS STREAM variable=bo depth=1
		  #pragma HLS STREAM variable=bi depth=1
		  #pragma HLS STREAM variable=mo depth=1
		  #pragma HLS STREAM variable=mi depth=1
		  #pragma HLS STREAM variable=Rwo depth=1
		  #pragma HLS STREAM variable=Rwi depth=1
	  }

	  // Multiply bo = Ro^T X
	  // Multiply bi = Ri^T X
	  for(int ii = 0; ii < CONFIG_T::n_edge; ii++)
	  {
		  if (CONFIG_T::io_type == io_serial)
		  {
			  #pragma HLS PIPELINE
		  }
		  for(int jj = 0; jj < CONFIG_T::n_input_dim; jj++)
		  {
			  bi[ii][jj] = 0;
			  bo[ii][jj] = 0;
			  for(int kk = 0; kk < CONFIG_T::n_node; kk++)
			  {
				  bi[ii][jj] += Ri[kk][ii] * X[kk][jj];
				  bo[ii][jj] += Ro[kk][ii] * X[kk][jj];
			  }
		  }
	  }

	  // Hadamard Multiply Rwo = Ro * e
	  // Hadamard Multiply Rwi = Ri * e
	  for(int ii = 0; ii < CONFIG_T::n_node; ii++)
	  {
		  if (CONFIG_T::io_type == io_serial)
		  {
			  #pragma HLS PIPELINE
			  }
		  for(int jj = 0; jj < CONFIG_T::n_edge; jj++)
		  {
			  Rwo[ii][jj] = Ro[ii][jj] * e[jj][0];
			  Rwi[ii][jj] = Ri[ii][jj] * e[jj][0];
		  }
	  }

	  // Multiply mi = Rwi bo
	  // Multiply mo = Rwo bi
	  int counter = 0;
	  for(int ii = 0; ii < CONFIG_T::n_node; ii++)
	  {
		  if (CONFIG_T::io_type == io_serial)
		  {
			  #pragma HLS PIPELINE
			  }
		  for(int jj = 0; jj < CONFIG_T::n_input_dim; jj++)
		  {
			  mi[ii][jj] = 0;
			  mo[ii][jj] = 0;
			  for(int kk = 0; kk < CONFIG_T::n_edge; kk++)
			  {
				  counter+=1;
				  mi[ii][jj] += Rwi[ii][kk] * bo[kk][jj];
				  mo[ii][jj] += Rwo[ii][kk] * bi[kk][jj];
			  }
			  // Concatenate M = [mi, mo, X]
			  M[ii][jj] = (res_T) mi[ii][jj];
			  M[ii][CONFIG_T::n_input_dim+jj] = (res_T) mo[ii][jj];
			  M[ii][2*CONFIG_T::n_input_dim+jj] = (res_T) X[ii][jj];
		  }
	  }
  }
}

#endif
