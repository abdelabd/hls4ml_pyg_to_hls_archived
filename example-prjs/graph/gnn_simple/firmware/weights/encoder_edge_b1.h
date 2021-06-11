//Numpy array shape [8]
//Min 0.005000714561
//Max 0.160120687321
//Number of zeros 0

#ifndef ENCODER_EDGE_B1_H_
#define ENCODER_EDGE_B1_H_

#ifndef __SYNTHESIS__
ap_fixed<16,6> encoder_edge_b1[8];
#else
ap_fixed<16,6> encoder_edge_b1[8] = {0.160121, 0.005001, 0.110611, 0.065002, 0.140449, 0.056003, 0.118520, 0.125007};
#endif

#endif
