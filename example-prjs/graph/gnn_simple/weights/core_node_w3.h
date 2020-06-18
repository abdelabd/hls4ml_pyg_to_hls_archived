//Numpy array shape [1, 1]
//Min 0.0
//Max 0.0
//Number of zeros 1

#ifndef CORE_NODE_W3_H_
#define CORE_NODE_W3_H_

#ifndef __SYNTHESIS__
ap_fixed<16,6> core_node_w3[1];
#else
ap_fixed<16,6> core_node_w3[1] = {0.0};
#endif

#endif
