//Numpy array shape [8]
//Min -0.247301383978
//Max 0.310068572518
//Number of zeros 0

#ifndef DECODER_EDGE_B0_H_
#define DECODER_EDGE_B0_H_

#ifndef __SYNTHESIS__
ap_fixed<16,6> decoder_edge_b0[8];
#else
ap_fixed<16,6> decoder_edge_b0[8] = {-0.099610, -0.201869, -0.057497, 0.266779, 0.310069, -0.247301, 0.219399, 0.088466};
#endif

#endif
