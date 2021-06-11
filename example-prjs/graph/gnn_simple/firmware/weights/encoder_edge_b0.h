//Numpy array shape [8]
//Min 0.004226764008
//Max 0.177938968798
//Number of zeros 0

#ifndef ENCODER_EDGE_B0_H_
#define ENCODER_EDGE_B0_H_

#ifndef __SYNTHESIS__
ap_fixed<16,6> encoder_edge_b0[8];
#else
ap_fixed<16,6> encoder_edge_b0[8] = {0.024553, 0.177939, 0.018872, 0.019578, 0.004227, 0.132558, 0.075681, 0.105626};
#endif

#endif
