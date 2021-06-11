//Numpy array shape [8]
//Min -0.912902561424
//Max 0.258008742093
//Number of zeros 1

#ifndef DECODER_EDGE_B2_H_
#define DECODER_EDGE_B2_H_

#ifndef __SYNTHESIS__
ap_fixed<16,6> decoder_edge_b2[8];
#else
ap_fixed<16,6> decoder_edge_b2[8] = {-0.007014, 0.000000, 0.258009, -0.316539, -0.361022, 0.053682, -0.912903, 0.099062};
#endif

#endif
