#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

//hls-fpga-machine-learning insert numbers
#define REUSE_GRAPH 8
#define REUSE_DENSE 1
#define N_ITERS 1
#define LATENT_EDGE 8 //40
#define LATENT_NODE 8 //40
#define N_FEATURES 3
#define E_FEATURES 1 //4
//graph_nets simple example:
#define N_NODES_MAX 28 //28 112
#define N_EDGES_MAX 37 //37 148

#define PRAGMA_SUB(x) _Pragma (#x)
#define DO_PRAGMA(x) PRAGMA_SUB(x)

typedef ap_fixed<16,6> accum_default_t;
typedef ap_fixed<16,6> weight_default_t;
typedef ap_fixed<16,6> bias_default_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> result_t;
typedef ap_uint<16> index_t;

#endif
