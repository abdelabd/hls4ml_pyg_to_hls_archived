//Numpy array shape [8]
//Min -0.010451145761
//Max 0.246754384876
//Number of zeros 0

#ifndef ENCODER_NODE_B0_H_
#define ENCODER_NODE_B0_H_

#ifndef __SYNTHESIS__
ap_fixed<16,6> encoder_node_b0[8];
#else
ap_fixed<16,6> encoder_node_b0[8] = {0.101684, 0.075236, 0.085692, -0.010451, 0.188652, 0.013790, 0.246754, 0.080155};
#endif

#endif
