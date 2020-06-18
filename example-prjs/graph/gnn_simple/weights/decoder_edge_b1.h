//Numpy array shape [8]
//Min -0.323449562576
//Max 0.287279500500
//Number of zeros 0

#ifndef DECODER_EDGE_B1_H_
#define DECODER_EDGE_B1_H_

#ifndef __SYNTHESIS__
ap_fixed<16,6> decoder_edge_b1[8];
#else
ap_fixed<16,6> decoder_edge_b1[8] = {0.270760, -0.118365, -0.005526, 0.287280, 0.026505, -0.257067, -0.012887, -0.323450};
#endif

#endif
