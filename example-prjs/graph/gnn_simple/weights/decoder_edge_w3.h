//Numpy array shape [1, 8]
//Min -0.854861067400
//Max 1.621065692873
//Number of zeros 0

#ifndef DECODER_EDGE_W3_H_
#define DECODER_EDGE_W3_H_

#ifndef __SYNTHESIS__
ap_fixed<16,6> decoder_edge_w3[8];
#else
ap_fixed<16,6> decoder_edge_w3[8] = {-0.182186, -0.510035, 0.858173, -0.425018, -0.854861, -0.660627, 1.621066, -0.506390};
#endif

#endif
