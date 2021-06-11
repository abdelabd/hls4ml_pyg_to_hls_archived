#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 19:19:55 2021

@author: abdel
"""

import os 
import re
from collections import OrderedDict
import yaml
import torch
import numpy as np

#import hls4ml
#from hls4ml.model.hls_model import HLSModel
from hls4ml.utils.config import create_vivado_config

# locals
main_dir = "/home/abdel/IRIS_HEP/GNN_hls_layers"
os.chdir(main_dir)
#from pyg_to_hls import PygModelReader

# Model
from utils.models.interaction_network_pyg_2 import InteractionNetwork as InteractionNetwork_pyg
model = InteractionNetwork_pyg()
model_dict = torch.load("/home/abdel/IRIS_HEP/interaction_networks/trained_models/IN_pyg_2_small_state_dict.pt")
model.load_state_dict(model_dict)
del model_dict 

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

from examples import layer_list as layer_list_example
from examples import hls_model as hls_model_example
model_example = hls_model_example.reader.torch_model

#%%

def write_yml(config, save_dir):
    nonmodel_config = {}
    submodule_config = {}
    
    for key, val in config.items():
        if key == 'PytorchModel':
            for mkey, mval in val._modules.items():
                submodule_config[mkey] = mval      
        else:
            nonmodel_config[key] = val
    
    with open(save_dir+"/nonmodel_config.yml", 'w') as file:
        yaml.dump(nonmodel_config, file)
    with open(save_dir+"/submodule_config.yml", 'w') as file:
        yaml.dump(submodule_config, file)
    
    indent = "  "
    model_config_str = f"PytorchModel: !!python/object:__main__.{config['PytorchModel'].__class__.__name__}"
    
    model_config_str  += "\n" + indent + "_backward_hooks: !!python/object/apply:collections.OrderedDict"
    model_config_str += "\n" + indent + "- []"
    
    model_config_str  += "\n" + indent + "_buffers: !!python/object/apply:collections.OrderedDict"
    model_config_str  += "\n" + indent + "- []"
    
    model_config_str  += "\n" + indent + "_forward_hooks: !!python/object/apply:collections.OrderedDict"
    model_config_str  += "\n" + indent + "- []"
    
    model_config_str  += "\n" + indent + "_forward_pre_hooks: !!python/object/apply:collections.OrderedDict"
    model_config_str += "\n" + indent + "- []"
    
    model_config_str  += "\n" + indent + "_load_state_dict_pre_hooks: !!python/object/apply:collections.OrderedDict"
    model_config_str  += "\n" + indent + "- []"
    
    model_config_str  += "\n" + indent + "_non_persistent_buffer_sets: !!set {}"
    
    model_config_str  += "\n" + indent + "_parameters: !!python/object/apply:collections.OrderedDict"
    model_config_str  += "\n" + indent + "- []"

    model_config_str  += "\n" + indent + "_state_dict_hooks: !!python/object/apply:collections.OrderedDict"
    model_config_str  += "\n" + indent + "- []"
    
    model_config_str  += "\n" + indent + f"input_shape: {[m,q], [n,p], [2,m]}"
    model_config_str  += "\n" + indent + "quantized_model: false"
    model_config_str  += "\n" + indent + "training: true"
    
    
    with open(save_dir+"/submodule_config.yml", "r") as file:
        submodule_config_str = file.read()
        
    submodule_config_lines = submodule_config_str.split('\n')
    for i, line in enumerate(submodule_config_lines):
        colon_start = line.find(":")
        if line[:colon_start] in model._modules.keys():
            newline1 = "- - - " + line[:colon_start]
            newline2 = indent+indent+indent + "- "
            submodule_config_lines.insert(i)
            submodule_config_lines[i] = "- - - " + line[colon_start+2:]
            
    submodule_config_lines = [indent+i for i in submodule_config_lines]
    submodule_config_str = "\n".join(submodule_config_lines)
    
    top_config_str = model_config_str + "\n" + indent + "_modules: !!python/object/apply:collections.OrderdDict" + "\n"
    top_config_str += submodule_config_str
    
    with open(save_dir+"/nonmodel_config.yml", "r") as file:
        nonmodel_config_str = file.read()
    
    top_config_str = nonmodel_config_str + top_config_str
    with open(save_dir+"/hls4ml_config.yml", "w") as file:
        file.write(top_config_str)
         
    return top_config_str


#%%

top_config_str = write_yml(config, main_dir+"/write_yml_test")

#%%

with open(main_dir+"/write_yml_test/temp_config.yml", 'r') as file:
    a = file.read()

#%%
with open(main_dir+"/write_yml_test/str_to_yml_test.yml", 'w') as file:
    file.write(a)
            