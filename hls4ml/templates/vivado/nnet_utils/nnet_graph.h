#ifndef NNET_GRAPH_H_
#define NNET_GRAPH_H_

#include "nnet_common.h"
#include "nnet_merge.h"
#include "nnet_dense.h"
#include "nnet_dense_resource.h"
#include "nnet_activation.h"
#include "nnet_array.h"
#include <math.h>

namespace nnet {
  enum flow {source_to_target=0, target_to_source=1};
  enum aggr {aggr_sum=0, aggr_mean=1, aggr_max=2};
  
  struct graph_config
  {
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float table_t;
    
    // Layer Sizes
    static const unsigned n_node = 10;
    static const unsigned n_edge = 20;
    static const unsigned n_features = 3;
    static const unsigned e_features = 4;
    static const unsigned n_out = 4;
    static const unsigned n_layers = 3;

    // message-passing parameters
    static const unsigned aggr = aggr_sum;
    static const unsigned flow = source_to_target;
    
    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;

    static const bool activate_final = false;
  };

  struct aggregate_config
  {
     typedef float table_t;
     static const unsigned n_node = 10;
     static const unsigned n_edge = 20;
     static const unsigned edge_dim = 4;
     static const unsigned aggr = aggr_sum;
     static const unsigned flow = source_to_target;
  };

  // division-LUT for mean-aggregation
  inline float division(float input){
    return 1.0/input;
  }
  template<typename CONFIG_T, int N_TABLE>
  void init_div_table(typename CONFIG_T::table_t table_out[N_TABLE]){
    int j = 0;
    typename CONFIG_T::table_t k = 1;
    table_out[j] = k;
    for(int i=1; i<N_TABLE; i++){
      float in_val = float(i);
      typename CONFIG_T::table_t reciprocal = nnet::division(in_val);
      table_out[i] = reciprocal;
    }
  }
  template<class data_T, class index_T, class res_T, typename CONFIG_T>
  void edge_divide(data_T edge_sum_i, index_T n_edges_i, res_T &edge_mean_i){
    // initialize LUT
  #ifdef __HLS_SYN__
      bool initialized=false;
      typename CONFIG_T::table_t div_table[CONFIG_T::n_edge];
  #else
      static bool initialized=false;
      static typename CONFIG_T::table_t div_table[CONFIG_T::n_edge];
  #endif

      if(!initialized){
        nnet::init_div_table<CONFIG_T, CONFIG_T::n_edge>(div_table);
        initialized=true;
      }

      if(CONFIG_T::io_type==io_parallel){
        #pragma HLS PIPELINE
      }

      data_T reciprocal;
      reciprocal = div_table[n_edges_i];
      edge_mean_i = edge_sum_i*reciprocal;
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void dense_mult_1lyr(
			 data_T data[CONFIG_T::dense_config1::n_in],
			 res_T res[CONFIG_T::dense_config1::n_out],
			 typename CONFIG_T::dense_config1::weight_t weights0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			 typename CONFIG_T::dense_config1::bias_t   biases0[CONFIG_T::dense_config1::n_out])
  {
    nnet::dense_resource<data_T, res_T, typename CONFIG_T::dense_config1>(data, res, weights0, biases0);
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
    nnet::dense_resource<data_T, data_T, typename CONFIG_T::dense_config1>(data, data0_logits, weights0, biases0);
    data_T data0[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config1>(data0_logits, data0);

    nnet::dense_resource<data_T, res_T, typename CONFIG_T::dense_config2>(data0, res, weights1, biases1);
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
    nnet::dense_resource<data_T, data_T, typename CONFIG_T::dense_config1>(data, data0_logits, weights0, biases0);
    data_T data0[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config1>(data0_logits, data0);

    data_T data1_logits[CONFIG_T::dense_config2::n_out];
    #pragma HLS ARRAY_PARTITION variable=data1_logits complete dim=0
    nnet::dense_resource<data_T, data_T, typename CONFIG_T::dense_config2>(data0, data1_logits, weights1, biases1);
    data_T data1[CONFIG_T::dense_config2::n_out];
    #pragma HLS ARRAY_PARTITION variable=data1 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config2>(data1_logits, data1);

    nnet::dense_resource<data_T, res_T, typename CONFIG_T::dense_config3>(data1, res, weights2, biases2);
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
    nnet::dense_resource<data_T, data_T, typename CONFIG_T::dense_config1>(data, data0_logits, weights0, biases0);
    data_T data0[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config1>(data0_logits, data0);

    data_T data1_logits[CONFIG_T::dense_config2::n_out];
    #pragma HLS ARRAY_PARTITION variable=data1_logits complete dim=0
    nnet::dense_resource<data_T, data_T, typename CONFIG_T::dense_config2>(data0, data1_logits, weights1, biases1);
    data_T data1[CONFIG_T::dense_config2::n_out];
    #pragma HLS ARRAY_PARTITION variable=data1 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config2>(data1_logits, data1);

    data_T data2_logits[CONFIG_T::dense_config3::n_out];
    #pragma HLS ARRAY_PARTITION variable=data2_logits complete dim=0
    nnet::dense_resource<data_T, data_T, typename CONFIG_T::dense_config3>(data1, data2_logits, weights2, biases2);
    data_T data2[CONFIG_T::dense_config3::n_out];
    #pragma HLS ARRAY_PARTITION variable=data2 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config3>(data2_logits, data2);

    nnet::dense_resource<data_T, res_T, typename CONFIG_T::dense_config4>(data2, res, weights3, biases3);
  }

  template<class data_T, class index_T, class res_T, typename CONFIG_T>
    void aggregate(
            data_T    edge_attr_1D[CONFIG_T::n_edge*CONFIG_T::edge_dim],
            index_T   edge_index_1D[CONFIG_T::n_edge*2],
            res_T     edge_attr_aggr_1D[CONFIG_T::n_node*CONFIG_T::edge_dim])
  {
    //initialize arrays
    // 1. edge_attr (input)
    data_T edge_attr[CONFIG_T::n_edge][CONFIG_T::edge_dim];
    #pragma HLS ARRAY_PARTITION variable=edge_attr complete dim=0
    nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::edge_attr_config>(edge_attr_1D, edge_attr);

    // 2. edge_index (input)
    index_T edge_index[CONFIG_T::n_edge][2];
    #pragma HLS ARRAY_PARTITION variable=edge_index complete dim=0
    nnet::vec_to_mat<index_T, index_T, typename CONFIG_T::edge_index_config>(edge_index_1D, edge_index);

    //3. num_edge_per_node (intermediate)
    index_T num_edge_per_node[CONFIG_T::n_node];
    for(int i=0; i<CONFIG_T::n_node; i++){
      num_edge_per_node[i]=0;
    }

    //4. edge_attr_aggr (output)
    res_T edge_attr_aggr[CONFIG_T::n_node][CONFIG_T::edge_dim];
    #pragma HLS ARRAY_PARTITION variable=edge_update_aggr complete dim=0
    for(int i=0; i < CONFIG_T::n_node; i++){
      for(int j=0; j<CONFIG_T::edge_dim; j++){
        #pragma HLS UNROLL
        edge_attr_aggr[i][j] = 0;
      }
    }

    int sender_col;
    int receiver_col;
    if(CONFIG_T::flow == source_to_target){
      receiver_col = 1;
    }
    else{
      receiver_col = 0;
    }

    for(int i=0; i<CONFIG_T::n_edge; i++){
      index_T r = edge_index[i][receiver_col];
      num_edge_per_node[r] += 1;

      if((CONFIG_T::aggr == aggr_sum)||(CONFIG_T::aggr==aggr_mean)){
        for(int j=0; j<CONFIG_T::out_dim; j++){
          #pragma HLS UNROLL
          edge_attr_aggr[r][j] += edge_attr[i][j];
        }
      }
      else{ //CONFIG_T::aggr==aggr_max
        for(int j=0; j<CONFIG_T::out_dim; j++){
          #pragma HLS UNROLL
          edge_attr_aggr[r][j] = edge_attr[i][j];
        }
      }
    }

    // sum --> mean
    if(CONFIG_T::aggr == aggr_mean){
      for(int i=0; i < CONFIG_T::n_node; i++){
        for (int j=0; j<CONFIG_T::out_dim; j++){
          res_T edge_mean_j;
          nnet::edge_divide<res_T, index_T, res_T, CONFIG_T>(edge_attr_aggr[i][j], num_edge_per_node[i], edge_mean_j);
          edge_attr_aggr[i][j] = edge_mean_j;
        }
      }
    }

    // None --> max
    if(CONFIG_T::aggr == aggr_max){ //note: the edge_update_aggr array has been initialized but IS NOT ZEROS
      for(int i=0; i < CONFIG_T::n_edge; i++){
        index_T r = edge_index[i][receiver_col];

        if(num_edge_per_node[r] > 1){ //if it has more than one edge, perform max-aggregation as usual
          for(int j=0; j<CONFIG_T::out_dim; j++){
            #pragma HLS UNROLL
            edge_attr_aggr[r][j] = edge_attr[i][j] > edge_attr_aggr[r][j] ? edge_attr[i][j] : edge_attr_aggr[r][j];
          }
        }
      }
    }

    //output array --> output vec
    nnet::mat_to_vec<res_T, res_T, typename CONFIG_T::edge_attr_aggr_config>(edge_attr_aggr, edge_attr_aggr_1D);
  }

  template<class data_T, class index_T, class res_T, typename CONFIG_T>
    void EdgeBlock(
            data_T    node_attr_1D[CONFIG_T::n_node*CONFIG_T::node_dim],
			data_T    edge_attr_1D[CONFIG_T::n_edge*CONFIG_T::edge_dim],
			index_T   edge_index_1D[CONFIG_T::n_edge*2],
			res_T     edge_update_1D[CONFIG_T::n_edge*CONFIG_T::out_dim],
			res_T     edge_update_aggr_1D[CONFIG_T::n_node*CONFIG_T::out_dim],
			typename CONFIG_T::dense_config1::weight_t  core_edge_w0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config1::bias_t    core_edge_b0[CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config2::weight_t  core_edge_w1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config2::bias_t    core_edge_b1[CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config3::weight_t  core_edge_w2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config3::bias_t    core_edge_b2[CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config4::weight_t  core_edge_w3[CONFIG_T::dense_config4::n_in*CONFIG_T::dense_config4::n_out],
			typename CONFIG_T::dense_config4::bias_t    core_edge_b3[CONFIG_T::dense_config4::n_out])
  {
    //initialize intermediate arrays: num_edge_per_node, edge_aggr_mask (only needed if aggr==max)
    index_T num_edge_per_node[CONFIG_T::n_node];
    index_T edge_aggr_mask[CONFIG_T::n_node];
    #pragma HLS ARRAY_PARTITION variable=num_edge_per_node complete dim=0
    for(int i=0; i<CONFIG_T::n_node; i++){
        num_edge_per_node[i] = 0;
        if(CONFIG_T::aggr==aggr_max){
          edge_aggr_mask[i] = 0;
        }
    }

    // initialize edge_update_aggr_1D with appropriate values
    if((CONFIG_T::aggr==aggr_sum)||(CONFIG_T::aggr==aggr_mean)){
      for(int i=0; i < CONFIG_T::n_node; i++){
        for(int j=0; j<CONFIG_T::out_dim; j++){
          #pragma HLS UNROLL
          edge_update_aggr_1D[i*CONFIG_T::out_dim+j] = 0;
        }
      }
    }
    else{ //CONFIG_T:aggr==aggr_max, we want to initialize this with the most negative number we can represent
      // note this only works for ap_ufixed types
      res_T most_negative_num;
      most_negative_num.V = 1;
      most_negative_num.V <<= res_T::width - 1;
      for(int i=0; i < CONFIG_T::n_node; i++){
        for(int j=0; j<CONFIG_T::out_dim; j++){
          #pragma HLS UNROLL
          edge_update_aggr_1D[i*CONFIG_T::out_dim+j] = most_negative_num;
        }
      }
    }

    int sender_col;
    int receiver_col;
    if(CONFIG_T::flow == source_to_target){
      sender_col = 0;
      receiver_col = 1;
    }
    else{
      sender_col = 1;
      receiver_col = 0;
    }

    //initialize indexers
    int edge_attr_start = 0 - CONFIG_T::edge_dim;
    int edge_update_start = 0 - CONFIG_T::out_dim;
    int edge_index_start = 0 - 2;
    int edge_update_aggr_start;

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    edge_loop: for(int i = 0; i < CONFIG_T::n_edge; i++) { //for each edge
      #pragma HLS UNROLL

      // update indexers
      edge_attr_start += CONFIG_T::edge_dim;
      edge_update_start += CONFIG_T::out_dim;
      edge_index_start += 2;

      // get edge attr
      data_T edge_attr_i[CONFIG_T::edge_dim];
      for(int j=0; j<CONFIG_T::edge_dim; j++){
        #pragma HLS UNROLL
        edge_attr_i[j] = edge_attr_1D[edge_attr_start+j];
      }

      // get sender, receiver indices
      index_T s = edge_index_1D[edge_index_start+sender_col];
      index_T r = edge_index_1D[edge_index_start+receiver_col];
      num_edge_per_node[r] += 1;
      if(CONFIG_T::aggr==aggr_max){
        edge_aggr_mask[r] = 1;
      }

      //get sender, receiver attr
      data_T node_attr_r[CONFIG_T::node_dim];
      data_T node_attr_s[CONFIG_T::node_dim];
      for(int j=0; j<CONFIG_T::node_dim; j++){
        #pragma HLS UNROLL
        node_attr_r[j] = node_attr_1D[r*CONFIG_T::node_dim+j];
        node_attr_s[j] = node_attr_1D[s*CONFIG_T::node_dim+j];
      }

      // construct NN input: <receiver, sender, edge>
      data_T node_concat[2*CONFIG_T::node_dim];
      #pragma HLS ARRAY_PARTITION variable=node_concat complete dim=0
      nnet::concatenate1d<data_T, data_T, data_T, typename CONFIG_T::merge_config1>(node_attr_r, node_attr_s, node_concat);
      data_T phi_input[CONFIG_T::edge_dim + 2*CONFIG_T::node_dim];
      #pragma HLS ARRAY_PARTITION variable=phi_input complete dim=0
      nnet::concatenate1d<data_T, data_T, data_T, typename CONFIG_T::merge_config2>(node_concat, edge_attr_i, phi_input);

      // send it through NN
      data_T edge_update_i[CONFIG_T::out_dim];
	  if(CONFIG_T::activate_final){
	    data_T edge_update_logits[CONFIG_T::out_dim];
        #pragma HLS ARRAY_PARTITION variable=edge_update_logits complete dim=0
        if(CONFIG_T::n_layers == 1){
	      nnet::dense_mult_1lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update_logits, core_edge_w0, core_edge_b0);
	      nnet::relu<data_T, res_T, typename CONFIG_T::relu_config1>(edge_update_logits, edge_update_i);
        }
        else if(CONFIG_T::n_layers == 2){
	      nnet::dense_mult_2lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update_logits, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1);
	      nnet::relu<data_T, res_T, typename CONFIG_T::relu_config2>(edge_update_logits, edge_update_i);
	    }
	    else if(CONFIG_T::n_layers == 3){
	      nnet::dense_mult_3lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update_logits, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2);
	      nnet::relu<data_T, res_T, typename CONFIG_T::relu_config3>(edge_update_logits, edge_update_i);
	    }
	    else if(CONFIG_T::n_layers == 4){
	      nnet::dense_mult_4lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update_logits, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
	      nnet::relu<data_T, res_T, typename CONFIG_T::relu_config4>(edge_update_logits, edge_update_i);
	    }
      }
      else{
        if(CONFIG_T::n_layers == 1){
	      nnet::dense_mult_1lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update_i, core_edge_w0, core_edge_b0);
          }
        else if(CONFIG_T::n_layers == 2){
	      nnet::dense_mult_2lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update_i, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1);
        }
        else if(CONFIG_T::n_layers == 3){
	      nnet::dense_mult_3lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update_i, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2);
        }
        else if(CONFIG_T::n_layers == 4){
	      nnet::dense_mult_4lyr<data_T, res_T, CONFIG_T>(phi_input, edge_update_i, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
        }
      }
      for(int j=0; j<CONFIG_T::out_dim; j++){
        #pragma HLS UNROLL
        edge_update_1D[edge_update_start+j] = edge_update_i[j];
      }

      // aggregation step
      edge_update_aggr_start = r*CONFIG_T::out_dim;
      if((CONFIG_T::aggr == aggr_sum)||(CONFIG_T::aggr==aggr_mean)){
        for(int j=0; j<CONFIG_T::out_dim; j++){
          #pragma HLS UNROLL
          edge_update_aggr_1D[edge_update_aggr_start+j] += edge_update_i[j];
        }
      }
      else{ //CONFIG_T::aggr==aggr_max
        for(int j=0; j<CONFIG_T::out_dim; j++){
          #pragma HLS UNROLL
          edge_update_aggr_1D[edge_update_aggr_start+j] = edge_update_i[j] > edge_update_aggr_1D[edge_update_aggr_start+j] ? edge_update_i[j] : edge_update_aggr_1D[edge_update_aggr_start+j];
        }
      }
    }

    // sum --> mean
    if(CONFIG_T::aggr == aggr_mean){
      edge_update_aggr_start = 0 - CONFIG_T::out_dim;
      for(int i=0; i < CONFIG_T::n_node; i++){
        edge_update_aggr_start += CONFIG_T::out_dim;
        for (int j=0; j<CONFIG_T::out_dim; j++){
          #pragma HLS UNROLL
          res_T edge_mean_j;
          nnet::edge_divide<res_T, index_T, res_T, CONFIG_T>(edge_update_aggr_1D[edge_update_aggr_start+j], num_edge_per_node[i], edge_mean_j);
          edge_update_aggr_1D[edge_update_aggr_start+j] = edge_mean_j;
        }
      }
    }

    // None --> max
    if(CONFIG_T::aggr == aggr_max){ //note: the edge_update_aggr array has been initialized but IS NOT ZEROS
      edge_update_aggr_start = 0 - CONFIG_T::out_dim;
      for(int i=0; i < CONFIG_T::n_node; i++){
        edge_update_aggr_start += CONFIG_T::out_dim;
        for(int j=0; j<CONFIG_T::out_dim; j++){
          #pragma HLS UNROLL
          edge_update_aggr_1D[edge_update_aggr_start+j] = edge_aggr_mask[i]*edge_update_aggr_1D[edge_update_aggr_start+j];
        }
      }
    }
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void NodeBlock(
			data_T    node_attr_1D[CONFIG_T::n_node*CONFIG_T::node_dim],
			data_T    edge_attr_aggr_1D[CONFIG_T::n_node*CONFIG_T::edge_dim],
			res_T     node_update_1D[CONFIG_T::n_node*CONFIG_T::out_dim],
			typename CONFIG_T::dense_config1::weight_t  core_node_w0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config1::bias_t    core_node_b0[CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config2::weight_t  core_node_w1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config2::bias_t    core_node_b1[CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config3::weight_t  core_node_w2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config3::bias_t    core_node_b2[CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config4::weight_t  core_node_w3[CONFIG_T::dense_config4::n_in*CONFIG_T::dense_config4::n_out],
			typename CONFIG_T::dense_config4::bias_t    core_node_b3[CONFIG_T::dense_config4::n_out])
  {
    //initialize indexers
    int node_start = 0 - CONFIG_T::node_dim;
    int edge_aggr_start = 0 - CONFIG_T::edge_dim;
    int node_update_start = 0 - CONFIG_T::out_dim;

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    node_loop: for(int i = 0; i < CONFIG_T::n_node; i++){ //for each node
      #pragma HLS UNROLL

      //update indexers
      node_start += CONFIG_T::node_dim;
      edge_aggr_start += CONFIG_T::edge_dim;
      node_update_start  += CONFIG_T::out_dim;

      //get node attr
      data_T node_attr_i[CONFIG_T::node_dim];
      for(int j=0; j<CONFIG_T::node_dim; j++){
        #pragma HLS UNROLL
        node_attr_i[j] = node_attr_1D[node_start+j];
      }
      //get edge_aggr attr
      data_T edge_aggr_i[CONFIG_T::edge_dim];
      for(int j=0; j<CONFIG_T::edge_dim; j++){
        #pragma HLS UNROLL
        edge_aggr_i[j] = edge_attr_aggr_1D[edge_aggr_start+j];
      }

      // construct NN input: <node, edge_attr_aggr>
      data_T phi_input[CONFIG_T::node_dim + CONFIG_T::edge_dim];
      #pragma HLS ARRAY_PARTITION variable=phi_input complete dim=0
      nnet::concatenate1d<data_T, data_T, data_T, typename CONFIG_T::merge_config1>(node_attr_i, edge_aggr_i, phi_input);

      // send it through NN
      res_T node_update_i[CONFIG_T::out_dim];
      if(CONFIG_T::activate_final){
	    data_T node_update_logits[CONFIG_T::node_dim];
	    #pragma HLS ARRAY_PARTITION variable=node_update_logits complete dim=0
	    if(CONFIG_T::n_layers == 1){
	      nnet::dense_mult_1lyr<data_T, data_T, CONFIG_T>(phi_input, node_update_logits, core_node_w0, core_node_b0);
	      nnet::relu<data_T, res_T, typename CONFIG_T::relu_config1>(node_update_logits, node_update_i);
	    }
	    else if(CONFIG_T::n_layers == 2){
	      nnet::dense_mult_2lyr<data_T, data_T, CONFIG_T>(phi_input, node_update_logits, core_node_w0, core_node_b0, core_node_w1, core_node_b1);
	      nnet::relu<data_T, res_T, typename CONFIG_T::relu_config2>(node_update_logits, node_update_i);
	    }
	    else if(CONFIG_T::n_layers == 3){
	      nnet::dense_mult_3lyr<data_T, data_T, CONFIG_T>(phi_input, node_update_logits, core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2);
	      nnet::relu<data_T, res_T, typename CONFIG_T::relu_config3>(node_update_logits, node_update_i);
	    }
	    else { // CONFIG_T::n_layers == 4
	      nnet::dense_mult_4lyr<data_T, data_T, CONFIG_T>(phi_input, node_update_logits, core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2, core_node_w3, core_node_b3);
	      nnet::relu<data_T, res_T, typename CONFIG_T::relu_config4>(node_update_logits, node_update_i);
	    }
      }
      else{
        if(CONFIG_T::n_layers == 1){
	      nnet::dense_mult_1lyr<data_T, res_T, CONFIG_T>(phi_input, node_update_i, core_node_w0, core_node_b0);
        }
        else if(CONFIG_T::n_layers == 2){
	      nnet::dense_mult_2lyr<data_T, res_T, CONFIG_T>(phi_input, node_update_i, core_node_w0, core_node_b0, core_node_w1, core_node_b1);
        }
        else if(CONFIG_T::n_layers == 3){
	      nnet::dense_mult_3lyr<data_T, res_T, CONFIG_T>(phi_input, node_update_i, core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2);
        }
        else { // CONFIG_T::n_layers == 4
	      nnet::dense_mult_4lyr<data_T, res_T, CONFIG_T>(phi_input, node_update_i, core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2, core_node_w3, core_node_b3);
        }
      }
      for(int j=0; j<CONFIG_T::out_dim; j++){
        #pragma HLS UNROLL
        node_update_1D[node_update_start+j] = node_update_i[j];
      }
    }
  }

}

#endif
