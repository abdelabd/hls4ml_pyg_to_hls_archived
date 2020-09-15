//Numpy array shape [32, 32]
//Min -2.760336404415
//Max 1.395009098018
//Number of zeros 0

#ifndef ENCODER_NODE_W1_H_
#define ENCODER_NODE_W1_H_

#ifndef __SYNTHESIS__
ap_fixed<16,6> encoder_node_w1[1024];
#else
ap_fixed<16,6> encoder_node_w1[1024] = {-0.199635, 0.182055, 0.026487, -0.014926, -0.228043, 0.027503, 0.122788, -0.117238, 0.018051, -0.182521, 0.031991, 0.092989, -0.519502, 0.044989, -0.337259, 0.134836, 0.165769, -0.225937, -0.076857, -0.210565, -0.103766, -0.090735, -0.219577, 0.124828, 0.160676, -0.165372, -0.070930, 0.148172, -0.306094, -0.341156, 0.139313, -0.301674, -0.231273, 0.267202, 0.122572, 0.069988, -0.214513, 0.077625, 0.045957, -0.240372, 0.064508, -0.159700, -0.118196, -0.239278, -0.549801, -0.007706, -0.353339, 0.005425, 0.073262, 0.074198, -0.279299, 0.091613, 0.060073, -2.091345, -0.302231, -0.175240, -0.202476, -0.182250, 0.157179, 0.280738, 0.189773, -0.686454, -0.239523, -0.180600, -0.213225, -0.055788, 0.407495, -0.130691, 0.257492, -0.132323, 0.454075, 0.071264, -0.268422, 0.070507, -0.198452, 0.123127, 0.395605, 0.019455, -0.226974, -0.320881, 0.182870, -0.292161, 0.118001, -0.236389, -0.241821, 0.942186, -0.136958, 0.338307, -0.561463, 0.162900, 0.099549, -0.433748, 0.014345, 0.016108, 0.176798, -0.125402, 0.280574, 0.039652, -0.172825, -0.710138, 0.295259, -0.176779, -0.273742, 0.003047, 0.067354, -0.247438, 0.401567, -0.467742, 0.180676, 0.331907, 0.182319, 0.103378, -0.149733, 0.224859, -0.635057, -0.052406, 0.395532, 0.131708, -0.138587, 0.081554, 0.103329, -0.163746, -0.434111, 0.087543, -0.081204, 0.119079, 0.015031, 0.054257, -0.007386, -0.080769, 0.071347, 0.163514, 0.072723, -1.301680, 0.100363, -0.154049, -0.097373, 0.198147, -0.166882, 0.253412, 0.180707, 0.027649, 0.126934, 0.239915, -0.244873, 0.216294, 0.144594, -0.044233, -0.342223, 0.039044, 0.235993, -0.241968, -0.029153, 0.017707, -0.120052, -0.653373, -0.692992, -0.218049, 0.112720, 0.012376, -0.000511, -0.400890, 0.295043, 0.000328, 0.077227, -0.397359, 0.023406, -0.114847, 0.159253, 0.129526, 0.171833, 0.619292, 0.022613, -0.070392, -0.224181, 0.086525, 0.125669, -0.048621, -0.297568, 0.132155, 0.206521, -0.528620, -0.528512, 0.009265, -0.250320, -0.312477, 0.155665, -0.251552, -0.037490, 0.124642, 0.146311, 0.016677, 0.129380, -0.392743, -0.170910, 0.150819, -0.078340, 0.019464, -0.146354, 0.204579, -0.111613, -0.293462, 0.220672, -0.661591, 0.421812, 0.139339, -0.032254, 0.057796, -0.172825, -0.087529, -0.467481, -0.149024, 0.034815, -0.613111, -0.232754, 0.132363, 0.286891, -0.068020, 0.257738, 0.276116, -0.298041, -0.026405, 0.194722, -0.257926, -0.126918, 0.048555, 0.291111, -0.263931, -0.043857, -0.117069, 0.227953, -0.230984, 0.087623, 0.113019, 0.044277, 0.037073, 0.060234, -0.226005, 0.160948, 0.165485, -0.290576, -0.003478, -0.039013, -0.330715, -0.139409, -0.002272, -0.162254, 0.364737, -0.108842, -0.171231, 0.207280, -0.135278, -0.039629, -0.078373, -0.133624, -0.068379, -0.208983, -0.188915, 0.216944, 0.141032, 0.000740, -0.336399, 0.012851, 0.005151, -0.147541, 0.047783, 0.059010, -0.035626, 0.048768, 0.141284, -0.187390, -0.240532, 0.079257, 0.178922, 0.229777, 0.050142, -0.526819, -0.663095, 0.122703, -0.096165, 0.222504, -0.388563, -0.239427, 0.110750, -0.229486, -0.249258, -0.144293, 0.088504, 0.192604, 0.113046, 0.246504, 0.040140, 0.210097, -0.509587, 0.142746, -0.007981, -0.207627, -0.233911, 0.003147, 0.008002, -0.043401, 0.012268, -0.385108, -0.074896, -0.024401, -0.612443, -0.025591, -0.206774, -0.548366, 0.078019, 0.120407, 0.200863, 0.019481, -0.204445, -0.234601, 0.047199, -0.056170, -0.268996, -0.015211, 0.012667, 0.318838, 0.133118, 0.156669, -0.191187, 0.335484, 0.248158, -0.293112, -0.040496, 0.073279, -0.054254, 0.072110, -0.367606, -0.230523, -0.166168, -0.053865, 0.065130, -0.196239, 0.071664, -0.922607, -0.274530, 0.146072, -0.190882, -0.604152, -0.292198, -0.252067, 0.015134, 0.029072, -0.165925, 0.354362, -0.351810, -0.303607, -0.476655, -0.079854, -0.337183, -0.131295, -0.147201, 0.312840, -0.271431, -0.178060, -0.133707, 0.023941, -0.117460, 0.047416, -0.223099, 0.253650, 0.028907, 0.023448, 0.059585, 0.097170, 0.236704, -0.200523, -0.041381, -0.216214, -1.918217, -0.128853, 0.024883, -0.136018, -0.583120, 0.520026, 0.016044, 0.225213, 0.070618, -0.176245, 0.227584, 0.220920, 0.049648, 0.121572, -0.330442, -0.097245, 0.045078, -0.317850, -0.217783, 0.336181, -0.034306, 0.025502, -0.051164, -0.261269, -0.227178, -0.416403, -0.114272, -1.875911, -0.088078, -0.289223, -0.452490, 0.045580, 0.071593, 0.018268, 0.049419, -0.054275, -0.130281, -1.181592, 0.019209, 0.923980, -0.299648, -0.061405, -0.180442, 0.086433, 0.240232, 0.065080, 0.185252, 0.021220, -0.037455, -0.386518, 0.188815, 0.352908, 0.160706, 0.214837, -0.184555, 0.233019, -0.248961, 0.215706, 0.132144, 0.000849, 0.052041, -0.256630, -0.046612, -0.371036, -0.222296, -0.020133, 0.115340, -0.273272, -0.312269, -0.165927, -0.008353, 0.200318, 0.337870, 0.005227, 0.196553, 0.121487, -0.020297, -0.168620, -0.012477, -0.527480, 0.046910, -0.081502, 0.028579, 0.045562, -0.208356, 0.137024, -0.239309, -0.245513, 0.148813, 0.062808, 0.139207, 0.230960, -0.097039, -1.102020, 0.094307, -0.047560, -0.242798, -0.064551, -0.147534, 0.087947, 0.243137, -0.061097, -0.315444, -0.040013, 0.225228, -0.065069, -0.356008, -0.184530, -1.720763, -0.279870, -0.388532, -0.061869, -0.125877, -0.002563, 0.171936, 0.103432, -0.165042, 0.131804, 0.032385, -0.212975, 0.141395, -0.034318, 0.127888, -0.208430, -0.157757, 0.264552, 0.160703, -0.569978, -0.033197, -0.365949, -0.078887, 0.028550, 0.033221, -0.176740, -0.021900, -0.465000, -0.061576, -0.045853, -0.228327, 0.514612, -0.005628, 0.320665, -0.069541, 0.575009, -0.162569, 0.212351, -0.112791, 0.759190, 0.235180, 1.239074, -0.027837, 0.337035, -1.166167, -0.248464, 1.395009, -0.137267, 0.795571, -0.320092, 0.180246, 0.561383, 0.253910, 0.358001, -0.257745, -0.576636, 0.839843, -1.620288, -0.616869, 0.296978, 0.032406, 0.148187, -0.047405, 0.110220, -0.086706, -0.236876, 0.116393, 0.196652, -0.253664, -0.163788, 0.099152, -0.187756, -0.253186, 0.164039, -0.039522, 0.034487, -0.027665, -0.462580, -0.307330, -1.293568, -0.003179, -0.059343, -0.100983, -0.213873, 0.010811, -0.071244, 0.174750, -0.220376, 0.003430, 0.043310, 0.290536, 0.077320, 0.124281, -0.019056, -0.344338, 0.134790, -0.131282, -0.212614, -0.025131, 0.056372, -0.024503, 0.105232, 0.039611, 0.150516, -0.050122, -0.087855, 0.214433, -0.110071, -0.108495, -0.050050, -0.144974, -0.805047, 0.044541, 0.287747, -0.165118, -0.826437, 0.260935, 0.099158, 0.085384, -0.129949, -0.193748, -0.009902, -0.230778, -0.104694, 0.064628, -0.159383, -0.271771, 0.156151, -0.086498, 0.161075, 0.088714, 0.105139, -0.087130, 0.005921, 0.015040, -0.005244, -0.051014, -0.200672, -0.296795, 0.273203, -0.267598, -0.110403, 0.111871, 0.078372, 0.119376, -0.715339, 0.046689, -0.238196, -1.393359, 0.075999, 0.080188, 0.013964, 0.223241, -0.254175, 0.036770, -0.279561, -0.255245, 0.194834, 0.033299, 0.062854, 0.005477, -0.144334, 0.084014, 0.328432, -0.121392, -0.063498, 0.145463, -0.235565, -0.231222, 0.251073, -0.479341, 0.149123, 0.084660, 0.131708, -0.022642, -0.129326, 0.016707, -0.564419, 0.133604, -0.175323, 0.191846, 0.218000, -0.365873, -0.068384, 0.228458, -0.224251, -0.743411, 0.018587, 0.197701, 0.106228, 0.166250, 0.075746, 0.265882, -0.131189, 0.265443, 0.345929, -0.642013, -0.190514, -0.258254, -0.234562, -0.154748, -0.358376, 0.140510, 0.318875, -0.210909, 0.378757, -0.083804, -0.368278, 0.110170, 0.209358, 0.433093, 0.460959, 0.037178, 0.179803, -0.082704, -0.172675, 0.396204, -0.299010, 0.008978, 0.061545, 0.280100, -0.342577, 0.372170, -0.166056, -2.207547, 0.247444, -0.031026, 0.390550, -0.141125, -0.050750, 0.144457, -0.038859, 0.057343, -0.015939, -0.221221, -0.059066, 0.261987, 0.144473, 0.096112, -0.188634, 0.025834, -0.648644, 0.023156, -0.210667, 0.201027, 0.099508, 0.192042, -0.358813, 0.339337, 0.047072, 0.009269, 0.289980, 0.130734, -0.015734, 0.064363, 0.164076, 0.436657, 0.194830, -0.121745, 0.128084, -0.782140, -0.219463, -0.201446, 0.021572, 0.043741, 0.011484, -0.185612, -0.128622, -0.259471, -0.176754, -0.068215, 0.164159, 0.109695, 0.282459, -0.168220, 0.107787, -0.340127, 0.138662, 0.046674, -0.047847, -0.048411, 0.154521, 0.164503, -0.721612, -0.086882, -0.201890, -0.021393, 0.007031, -0.123947, -0.143727, -0.225633, -0.200784, -0.227146, 0.250376, -0.138409, 0.171379, -0.002012, 0.043835, -0.605633, 0.108540, 0.256625, -0.032710, -0.157872, -0.182572, -0.003521, -0.426986, -0.142558, -0.694793, -1.271543, 0.138749, 0.345963, 0.075523, -0.345425, 0.135017, 0.167836, -0.023482, 0.191150, 0.154687, -0.162281, 0.025740, -2.760336, 0.097107, 0.098959, -0.227566, -0.001386, 0.120056, -0.033820, 0.207136, -0.160950, -0.133815, -0.122918, -0.270427, 0.214924, 0.117838, 0.076818, -0.363414, 0.234698, -0.138701, 0.361130, 0.112247, -0.079268, 0.294889, -0.115107, 0.091949, 0.233031, -0.065939, -0.165068, -0.159322, 0.137900, -0.202804, -0.033225, -0.365367, -0.022753, 0.047741, 0.254609, -0.399948, -0.242879, 0.346261, -0.084050, -0.209924, -0.226788, 0.192820, -0.036936, -0.240134, -0.051267, -0.072298, -0.325197, -0.096291, -0.169283, -0.105918, -0.073946, 0.170482, 0.026576, -0.342340, 0.245170, 0.182991, 0.153387, -0.173223, -0.019700, -0.253914, 0.037543, -0.137550, 0.141801, 0.334587, 0.177230, -0.401402, 0.062601, 0.175698, -1.455668, -0.900401, -0.212556, 0.076340, 0.044889, -0.140966, -0.176410, 0.322949, -0.428874, 0.509527, -0.185706, -0.372318, 0.421132, -0.140665, -0.214397, 0.354379, 0.296093, -0.199364, -0.177771, -0.324723, 0.020970, -0.145009, -0.022108, 0.312795, 0.253383, -0.088387, 0.551498, 0.095104, -0.195685, 0.340897, 0.001316, 0.036548, -0.368942, -0.069849, -0.245377, -0.043622, -0.351603, 0.199946, -0.102080, 0.247104, -0.004676, -0.017395, -0.115571, -0.042313, -0.251210, 0.225291, 0.229574, -0.161030, -0.304583, 0.047298, 0.057480, 0.076071, -0.418418, -0.095789, 0.174964, -0.133392, 0.217416, -0.156664, 0.106380, -0.157799, 0.037082, 0.089160, -0.265092, -0.648599, 0.062885, 0.158246, -0.120580, 0.002105, 0.199402, -0.202673, 0.133874, -0.095430, -0.214170, -0.022180, 0.164738, -0.049303, 0.009025, -0.038617, 0.053088, 0.002364, -0.199363, -0.134935, -0.021117, -0.110428, 0.315289, 0.008158, -0.540988, 0.337810, -0.150725, 0.105953, -0.301227, -0.372630, -0.742964, -0.118490, -0.065693, -0.068023, -0.231494, -0.338663, -0.025895, -0.124965, 0.091922, 0.043531, -0.092244, -0.072802, -0.182617, -0.127872, 0.171371, -0.029018, -0.292362, -0.319962, -0.318297, -0.180710, -0.536831, -0.375393, -0.083356, 0.244539, -0.396618, -0.176636, -0.216560, 0.155822, 0.041757, 0.148620, -0.081256, -0.052296, -0.273892, 0.087545, -0.331307, -0.151412, -0.073566, 0.006481, 0.142971, 0.182743, 0.032683, -0.299146, -0.207197, -0.131548, -0.447761, -0.122613, 0.133382, -0.329734, -0.256917, -0.111231, -0.076386, -0.263239, -0.234379, -0.603242, 0.048571, 0.097876, -0.082649, 0.231274, 0.150990, 0.203970, -0.131540, 0.203202, -0.065313};
#endif

#endif
