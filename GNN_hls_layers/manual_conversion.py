#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 15:26:23 2021

@author: abdel
"""

import os 
import re
from collections import OrderedDict
import torch
import numpy as np

import hls4ml
from hls4ml.model.hls_model import HLSModel
from hls4ml.utils.config import create_vivado_config

# locals
main_dir = "/home/abdel/IRIS_HEP/GNN_hls_layers"
os.chdir(main_dir)
from pyg_to_hls import PygModelReader

# Model
from utils.models.interaction_network_pyg import InteractionNetwork as InteractionNetwork_pyg
model = InteractionNetwork_pyg()
model_dict = torch.load("/home/abdel/IRIS_HEP/interaction_networks/trained_models/IN_pyg_small_state_dict.pt")
model.load_state_dict(model_dict)
del model_dict 

from examples import layer_list as layer_list_example
from examples import hls_model as hls_model_example

model_example = hls_model_example.reader.torch_model

#%%

n = 112 # num_nodes
m = 148 # num_edges
p = 3 # node_dim
q = 4 # edge_dim
r = 4 # effect_dim

config = {
    "output_dir": main_dir+"/hls_output",
    "project_name": "myproject",
    "fpga_part": 'xcku115-flvb2104-2-i',
    "clock_period": 5,
    "io_type": "io_parallel"
    }
config = create_vivado_config(**config)
config['PytorchModel'] = model
config['n_nodes'] = n
config['n_edges'] = m
config['node_dim'] = p
config['edge_dim'] = q

model_config = {
    'Precision': 'ap_fixed<16,6>',
    'ReuseFactor': 1,
    'Strategy': 'Latency'
    }

config['HLSConfig']['Model'] = model_config

layer_list = []
reader = PygModelReader(config)
input_shapes = reader.input_shapes
output_shapes = {}

EdgeAttr_layer = {
    'name': 'Re',
    'class_name': 'InputLayer',
    'input_shape': input_shapes['EdgeAttr'],
    'inputs': 'input'
    }
layer_list.append(EdgeAttr_layer)

NodeAttr_layer = {
    'name': 'Rn',
    'class_name': 'InputLayer',
    'input_shape': input_shapes['NodeAttr'],
    'inputs': 'input'
    }
layer_list.append(NodeAttr_layer)

EdgeIndex_layer = {
    'name': 'edge_index',
    'class_name': 'InputLayer',
    'input_shape': input_shapes['EdgeIndex'],
    'inputs': 'input'
    }
layer_list.append(EdgeIndex_layer)

R1_layer = {
    'name': 'R1',
    'class_name': 'EdgeBlock',
    'n_node': n,
    'n_edge': m,
    'n_features': p,
    'e_features': q,
    'out_features': q,
    'inputs': ['Re', 'Rn', 'edge_index'],
    'outputs': ["layer4_out_L", "layer4_out_Q"]
    }
layer_list.append(R1_layer)

O_layer = {
    'name': 'O',
    'class_name': 'NodeBlock',
    'n_node': n,
    'n_edge': m,
    'n_features': p,
    'e_features': q,
    'out_features': p,
    'inputs': ['Rn', "layer4_out_Q"],
    'outputs': ["layer5_out_P"]
    }
layer_list.append(O_layer)

R2_layer = {
    'name': 'R2',
    'class_name': 'EdgeBlock',
    'n_node': n,
    'n_edge': m,
    'n_features': p,
    'e_features': q,
    'out_features': 1,
    'inputs': ['layer4_out_L', 'layer5_out_P', 'edge_index'],
    'outputs': ['layer6_out_L', 'layer6_out_Q']
    }
layer_list.append(R2_layer)


class HLSModel_wrapper(HLSModel):
    def __init__(self, config, reader, layer_list):
        super().__init__(config, reader, layer_list)
            
            
    def get_weights_data(self, module_name, layer_name, var_name):
        return self.reader.get_weights_data(module_name, layer_name, var_name)
    
hls_model = HLSModel_wrapper(config, reader, layer_list)
hls_model.inputs = ['Re', 'Rn', 'edge_index']
hls_model.outputs = ['layer6_out_L']

#%% 

hls_model.compile()

#%%

hls_model.build()

#%%
jawn = {
    'R1': model.R1,
    'O': model.O,
    'R2': model.R2
    }
hls_model.config.config['PytorchModel'] = jawn
hls_model.compile()

#%%
import yaml
a = hls_model.config.config
filepath = hls_model.config.get_output_dir() + '/hls4ml_config.yml' 
with open(filepath, 'w') as file:
    yaml.dump(a, file)
    
#%%
b = hls_model_example.config.config
filepath_b = '/home/abdel/IRIS_HEP/javiers_example/hls_output/MLP_w_batnorm/hls4ml_config.yml'
with open(filepath_b, 'w') as file:
    yaml.dump(b, file)
    
#%%

a['PytorchModel'] = b['PytorchModel']
filepath = hls_model.config.get_output_dir() + '/hls4ml_config.yml' 
with open(filepath, 'w') as file:
    yaml.dump(a, file)
    
#%%

config = {"model": hls_model_example.reader.torch_model}
filepath = "/home/abdel/IRIS_HEP/GNN_hls_layers/yaml_dump_test/model_config.yml"
with open(filepath, 'w') as file:
    yaml.dump(config, file)

    