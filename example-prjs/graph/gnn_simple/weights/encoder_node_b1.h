//Numpy array shape [8]
//Min -0.313127790409
//Max 0.173565356787
//Number of zeros 0

#ifndef ENCODER_NODE_B1_H_
#define ENCODER_NODE_B1_H_

#ifndef __SYNTHESIS__
ap_fixed<16,6> encoder_node_b1[8];
#else
ap_fixed<16,6> encoder_node_b1[8] = {0.147346, 0.106749, 0.173565, -0.067655, -0.014823, -0.031195, 0.085009, -0.313128};
#endif

#endif
