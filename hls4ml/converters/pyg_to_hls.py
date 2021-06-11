from __future__ import print_function
import numpy as np
import os
import yaml
import sys
import torch
import pickle
import re
from hls4ml.model import HLSModel

from hls4ml.converters.pytorch_to_hls import register_pytorch_layer_handler, get_supported_pytorch_layers, pytorch_handler

#locals 
main_dir = "/home/abdel/IRIS_HEP/GNN_hls_layers"
os.chdir(main_dir)
from utils.layer_handlers import layer_handlers

#%%

class PygModelReader(object):
    
    def __init__(self, config):
        self.torch_model = config['PytorchModel']
        self.state_dict = self.torch_model.state_dict()
        self.n_nodes = config['n_nodes']
        self.n_edges = config['n_edges']
        self.node_dim = config['node_dim']
        self.edge_dim = config['edge_dim']
        self.input_shapes = {
            'EdgeAttr': [self.n_edges, self.edge_dim],
            'NodeAttr': [self.n_nodes, self.node_dim],
            'EdgeIndex': [2, self.n_edges]
            }
        
    def get_weights_data(self, module_name, layer_name, var_name):
        data = None
        
        #Parameter mapping from pytorch to keras
        torch_paramap = {
        #Conv
        'kernel': 'weight', 
        #Batchnorm
        'gamma': 'weight',
        'beta': 'bias',
        'moving_mean':'running_mean',
        'moving_variance': 'running_var'}
        
        if var_name not in list(torch_paramap.keys()) + ['weight', 'bias']:
            raise Exception('Pytorch parameter not yet supported!')
            
        elif var_name in list(torch_paramap.keys()):
            var_name = torch_paramap[var_name]
            
        try:
            data = self.state_dict[module_name + '.' + layer_name + '.' + var_name].numpy().transpose()
        except KeyError:
            data = self.state_dict[module_name + '.layers.' + layer_name + '.' + var_name].numpy().transpose()
            
        return data
        
#%%
