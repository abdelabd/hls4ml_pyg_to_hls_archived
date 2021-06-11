import os 
import re
import numpy as np 
from collections import OrderedDict
import torch 

import hls4ml
from hls4ml.model.hls_model import HLSModel
from hls4ml.model.hls_layers import Layer, layer_map, register_layer
from hls4ml.templates import templates

# Locals (for testing+development)
main_dir = "/home/abdel/IRIS_HEP/GNN_hls_layers"
os.chdir(main_dir)

# hls_model of simple MLP
from examples import layer_list, hls_model
graph = hls_model.graph

# torch model (InteractionNetwork from torch_geometric.nn.conv.MessagePassing)
from utils.models.interaction_network_pyg import InteractionNetwork as InteractionNetwork_pyg
model = InteractionNetwork_pyg()
model_dict = torch.load("/home/abdel/IRIS_HEP/interaction_networks/trained_models/IN_pyg_small_state_dict.pt")
model.load_state_dict(model_dict)
del model_dict 

phi_R1 = model.R1

#%% Templates 

EdgeBlock_function_template = templates.backend_map['Vivado'].get_function_template('EdgeBlock')
EdgeBlock_config_template = templates.backend_map['Vivado'].get_config_template('EdgeBlock')
EdgeBlock_include_list = templates.backend_map['Vivado'].get_include_list('EdgeBlock')

#%% class EdgeBlock

class EdgeBlock(Layer):
    def initialize(self):
        assert (len(self.inputs) == 3)  # edge_features, node_features, edge_index
        assert (len(self.outputs) == 2)  # edge_predictions, aggregated node_predictions

        self.n_node = self.attributes['n_node']
        self.n_edge = self.attributes['n_edge']
        self.n_features = self.attributes['n_features']
        self.e_features = self.attributes['e_features']
        self.out_features = self.attributes['out_features']
        self.torch_module = getattr(self.model.reader.torch_model, self.name)

        # edge predictions
        L_shape = [self.n_edge, self.out_features]
        L_dims = ['n_edge', f"layer{self.index}_out_features"]
        L_name = f"layer{self.index}_out_L"
        self.add_output_variable(shape=L_shape, dim_names=L_dims, out_name=L_name, var_name=L_name)

        # aggregated edge predictions (for use in NodeBlock)
        Q_shape = [self.n_node, self.out_features]
        Q_dims = ['n_node', f"layer{self.index}_out_features"]
        Q_name = f"layer{self.index}_out_Q"
        self.add_output_variable(shape=Q_shape, dim_names=Q_dims, out_name=Q_name, var_name=Q_name)

        self.add_weights(quantizer=self.get_attr('weight_quantizer'),
                         compression=self.model.config.get_compression(self))
        self.add_bias(quantizer=self.get_attr('weight_quantizer'))

    def function_cpp(self):
        params = {}
        params['config'] = 'config{}'.format(self.index)
        params['input_t'] = self.model.get_layer_output_variable('Re').type.name
        params['index_t'] = self.model.get_layer_output_variable('edge_index').type.name
        params['output_t'] = self.get_output_variable().type.name
        params['Re'] = self.model.get_layer_output_variable('Re').cppname
        params['Rn'] = self.model.get_layer_output_variable('Rn').cppname
        params['edge_index'] = self.model.get_layer_output_variable('edge_index').cppname
        params['L'] = f"layer{self.index}_out_L"
        params['Q'] = f"layer{self.index}_out_Q"

        params['w0'] = self.get_weights(f"{self.name}_w0")
        params['b0'] = self.get_weights(f"{self.name}_b0")
        params['w1'] = self.get_weights(f"{self.name}_w1")
        params['b1'] = self.get_weights(f"{self.name}_b1")
        params['w2'] = self.get_weights(f"{self.name}_w2")
        params['b2'] = self.get_weights(f"{self.name}_b2")
        params['w3'] = self.get_weights(f"{self.name}_w3")
        params['b3'] = self.get_weights(f"{self.name}_b3")

        out = self._function_template.format(**params)
        return out

    def config_cpp(self):
        top_params = self.get_EdgeBlock_params()
        top_config = self._config_template.format(**top_params)
        top_config = top_config.split('\n')[:-1]
        top_config = '\n'.join(top_config)

        sublayer_configs = self._config_sublayers()
        for layer, config in sublayer_configs.items():
            config = ['    ' + i for i in config.split('\n')]
            config = '\n'.join(config)

            top_config += '\n\n'
            top_config += config

        top_config += '\n};'
        return top_config

    def add_weights(self, quantizer=None, compression=False):
        linear_count = 0
        try:
            for name, module in self.torch_module.layers.named_modules():
                if module.__class__.__name__ == 'Linear':
                    data = self.model.get_weights_data(self.name, name, 'kernel')
                    var_name = f"{self.name}_w{linear_count}"
                    self.add_weights_variable(name=var_name, var_name=var_name, data=data, quantizer=quantizer,
                                              compression=compression)
                    linear_count += 1

            # DUMMIES
            if linear_count < 3:
                for i in range(linear_count, 4):
                    self.add_weights_variable(name=var_name, var_name=f"{self.name}_w{i}", data=data,
                                              quantizer=quantizer, compression=compression)

        except AttributeError:
            for name, module in self.torch_module.named_modules():
                if module.__class__.__name__ == 'Linear':
                    data = self.model.get_weights_data(self.name, name, 'kernel')
                    var_name = f"{self.name}_w{linear_count}"
                    self.add_weights_variable(name=var_name, var_name=var_name, data=data, quantizer=quantizer,
                                              compression=compression)
                    linear_count += 1

            # DUMMIES
            if linear_count < 3:
                for i in range(linear_count, 4):
                    self.add_weights_variable(name=var_name, var_name=f"{self.name}_w{i}", data=data,
                                              quantizer=quantizer, compression=compression)

    def add_bias(self, quantizer=None):
        precision = None
        type_name = None
        linear_count = 0

        try:
            for name, module in self.torch_module.layers.named_modules():
                if module.__class__.__name__ == 'Linear':
                    data = self.model.get_weights_data(self.name, name, 'bias')
                    var_name = f"{self.name}_b{linear_count}"
                    self.add_weights_variable(name=var_name, var_name=var_name, type_name=type_name, precision=precision,
                                              data=data, quantizer=quantizer)
                    linear_count += 1

            # DUMMIES
            if linear_count < 3:
                for i in range(linear_count, 4):
                    self.add_weights_variable(name=var_name, var_name=f"{self.name}_b{i}", type_name=type_name,
                                              precision=precision, data=data, quantizer=quantizer)

        except AttributeError:
            for name, module in self.torch_module.named_modules():
                if module.__class__.__name__ == 'Linear':
                    data = self.model.get_weights_data(self.name, name, 'bias')
                    var_name = f"{self.name}_b{linear_count}"
                    self.add_weights_variable(name=var_name, var_name=var_name, type_name=type_name, precision=precision,
                                              data=data, quantizer=quantizer)
                    linear_count += 1

            # DUMMIES
            if linear_count < 3:
                for i in range(linear_count, 4):
                    self.add_weights_variable(name=var_name, var_name=f"{self.name}_b{i}", type_name=type_name,
                                              precision=precision, data=data, quantizer=quantizer)

    def get_dense_params(self, dense_layer, linear_count):  # hard-coded for now
        params = {}
        params['type'] = 'dense'
        params['index'] = linear_count
        params['n_in'] = dense_layer.in_features
        params['n_out'] = dense_layer.out_features
        params['iotype'] = 'io_parallel'
        params['reuse'] = 1
        params['nzeros'] = 0
        params['accum_t'] = 'ap_fixed<16,6>'
        params['bias_t'] = 'ap_fixed<16,6>'
        params['weight_t'] = 'ap_fixed<16,6>'
        return params

    def get_relu_params(self, relu_count, last_n_out):  # hard-coded for now
        params = {}
        params['type'] = 'relu'
        params['index'] = relu_count
        params['n_in'] = last_n_out
        params['table_size'] = 1024
        params['iotype'] = 'io_parallel'
        return params

    def get_EdgeBlock_params(self):  # hard-coded for now
        params = {}
        params['type'] = 'edgeblock'
        params['index'] = self.index
        params['bias_t'] = 'ap_fixed<16,6>'
        params['weight_t'] = 'ap_fixed<16,6>'
        params['n_node'] = 112
        params['n_edge'] = 148
        params['n_in'] = 10
        params['n_hidden'] = 10
        params['n_out'] = 4
        params['n_layers'] = 3
        params['e_features'] = 4
        params['n_features'] = 3
        params['io_type'] = 'io_parallel'
        params['reuse'] = 1
        params['n_zeros'] = 0
        return params

    def config_layer(self, layer_type, layer_params):
        all_lines = self.model.config.backend.get_config_template(layer_type).split('\n')
        all_lines[0] = re.sub('struct config{index}', 'struct {type}_config{index}', all_lines[0])
        param_lines = []
        out = []

        for param in layer_params:
            p_lines = [i for i in all_lines if "{%s}" % param in i]
            if len(p_lines) == 1 and p_lines[0] not in param_lines:
                param_lines.append(p_lines[0])
            elif len(p_lines) < 1:
                print(f"param {param} not found in {layer_type} config template")
            else:
                print('damn')
                print(f"param: {param}")
                print(f"len(p_lines)={len(p_lines)}")
                print('')

        for line in all_lines:
            if line in param_lines:
                out.append(line)
            else:
                param_search = line.find('{')
                if param_search == -1:
                    out.append(line)

        out = '\n'.join(out)
        out = out.format(**layer_params)
        return out

    def _config_sublayers(self):
        linear_count = 0
        relu_count = 0
        configs = OrderedDict()

        for i, (name, module) in enumerate(self.torch_module.layers.named_modules()):
            if name == '':
                continue

            if isinstance(module, torch.nn.modules.linear.Linear):
                linear_count += 1
                layer_params = self.get_dense_params(module, linear_count)
                layer_config = self.config_layer('Dense', layer_params)
                configs[f"dense_config{linear_count}"] = layer_config
                last_n_out = layer_params['n_out']

            elif isinstance(module, torch.nn.modules.activation.ReLU):
                relu_count += 1
                layer_params = self.get_relu_params(relu_count, last_n_out)
                layer_config = self.config_layer('Activation', layer_params)
                configs[f"relu_config{relu_count}"] = layer_config
                last_n_out = layer_params['n_in']

        return configs
    
#%% independent methods

def get_dense_params(dense_layer, linear_count):
    params = {}
    params['type'] = 'dense'
    params['index'] = linear_count
    params['n_in'] = dense_layer.in_features
    params['n_out'] = dense_layer.out_features
    params['iotype'] = 'io_parallel'
    params['reuse'] = 1
    params['nzeros'] = 0
    params['accum_t'] = 'ap_fixed<16,6>'
    params['bias_t'] = 'ap_fixed<16,6>'
    params['weight_t'] = 'ap_fixed<16,6>'
    return params

def get_relu_params(relu_count, last_n_out):
    params = {}
    params['type'] = 'relu'
    params['index'] = relu_count
    params['n_in'] = last_n_out
    params['table_size'] = 1024
    params['iotype'] = 'io_parallel'
    return params

def get_EdgeBlock_params(index):
    params = {}
    params['type'] = 'edgeblock'
    params['index'] = index
    params['bias_t'] = 'ap_fixed<16,6>'
    params['weight_t'] = 'ap_fixed<16,6>'
    params['n_node'] = 112
    params['n_edge'] = 148
    params['n_in'] = 10
    params['n_hidden'] = 10
    params['n_out'] = 4
    params['n_layers'] = 3
    params['e_features'] = 4
    params['n_features'] = 3
    params['io_type'] = 'io_parallel'
    params['reuse'] = 1
    params['n_zeros'] = 0
    return params

def config_layer(layer_type, layer_params):
    all_lines = templates.backend_map['Vivado'].get_config_template(layer_type).split('\n')
    all_lines[0] = re.sub('struct config{index}', 'struct {type}_config{index}', all_lines[0])
    param_lines = []
    out = []
    
    for param in layer_params:
        p_lines = [i for i in all_lines if "{%s}"%param in i]
        if len(p_lines)==1:
            if p_lines[0] not in param_lines:
                param_lines.append(p_lines[0])
        elif len(p_lines)<1:
            print(f"param {param} not found in {layer_type} config template")
        else: print('damn')
    
    for line in all_lines:
        if line in param_lines:
            out.append(line)
        else:
            param_found = line.find('{')
            if param_found==-1:
                out.append(line)
    
    out = '\n'.join(out)
    out = out.format(**layer_params)
    return out

def config_sublayers(phi_block):
    linear_count = 0
    relu_count = 0
    configs = OrderedDict()
    
    for i, (name, module) in enumerate(phi_block.layers.named_modules()):
        if name=='':
            continue
        
        if isinstance(module, torch.nn.modules.linear.Linear):
            linear_count += 1
            layer_params = get_dense_params(module, linear_count)
            layer_config = config_layer('Dense', layer_params)
            configs[f"dense_config{linear_count}"] = layer_config
            last_n_out = layer_params['n_out']
            
        elif isinstance(module, torch.nn.modules.activation.ReLU):
            relu_count += 1
            layer_params = get_relu_params(relu_count, last_n_out)
            layer_config = config_layer('Activation', layer_params)
            configs[f"relu_config{relu_count}"] = layer_config
            last_n_out = layer_params['n_in']
            
    return configs

def config_cpp(phi_block, index):
    phi_block_params = get_EdgeBlock_params(index)
    phi_block_config = config_layer('EdgeBlock', phi_block_params)
    phi_block_config = phi_block_config.split('\n')[:-1]
    phi_block_config = '\n'.join(phi_block_config)
    
    sublayer_configs = config_sublayers(phi_block)
    for layer, config in sublayer_configs.items():
        config = ['    '+i for i in config.split('\n')]
        config = '\n'.join(config)
        
        phi_block_config += '\n\n'
        phi_block_config += config
        
    phi_block_config += "\n};"
    return phi_block_config

#%% Test

sublayer_configs = config_sublayers(phi_R1)  
phi_R1_params = get_EdgeBlock_params(1)
phi_R1_graph_config = config_layer('EdgeBlock', phi_R1_params)
phi_R1_full_config = config_cpp(phi_R1, 1)
        
        
    

    

    
