#!/usr/bin/env python
#import setGPU
import torch.nn as nn
import torch

import matplotlib.pyplot as plt
import torch.optim as optim
import math
import json
import yaml
import numpy as np
from datetime import datetime
import os
import os.path as path
from optparse import OptionParser

import hls4ml
from hls4ml.model.hls_model import HLSModel
from hls4ml.model.hls_layers import layer_map

# Locals
os.chdir('/home/abdel/IRIS_HEP/javiers_example')
#import jet_dataset
from utils import prep_for_hls

#%% Model

class three_layer_model_batnorm_masked(nn.Module):
    def __init__(self, bn_affine = True, bn_stats = True ):
        # Model with <16,64,32,32,5> Behavior
        super(three_layer_model_batnorm_masked, self).__init__()
        self.quantized_model = False
        self.input_shape = 16  # (16,)
        self.fc1 = nn.Linear(self.input_shape, 64)
        self.bn1 = nn.BatchNorm1d(64, affine=bn_affine, track_running_stats=bn_stats)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32, affine=bn_affine, track_running_stats=bn_stats)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 32)
        self.bn3 = nn.BatchNorm1d(32, affine=bn_affine, track_running_stats=bn_stats)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(32, 5)
        self.softmax = nn.Softmax(0)


    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.fc4(x)
        x = self.softmax(x)

        return x

dict_model = torch.load("32b_70Pruned_0rand.pth", map_location=lambda storage, loc: storage)
model = three_layer_model_batnorm_masked()
model.load_state_dict(dict_model)
del dict_model

#%% HLS Conversion

config, reader, layer_list = prep_for_hls(model, "MLP_w_batnorm", [None,16])
hls_model = HLSModel(config, reader, layer_list)
