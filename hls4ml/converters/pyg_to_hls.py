from __future__ import print_function

class PygModelReader(object):

    def __init__(self, config):
        self.torch_model = config['PytorchModel']
        self.state_dict = self.torch_model.state_dict()
        self.n_node = config['n_nodes']
        self.n_edge = config['n_edges']
        self.n_features = config['node_dim']
        self.e_features = config['edge_dim']
        self.input_shapes = {
            'EdgeAttr': [self.n_edge, self.e_features],
            'NodeAttr': [self.n_node, self.n_features],
            'EdgeIndex': [2, self.n_edge]
        }

    def get_weights_data(self, module_name, layer_name, var_name):
        data = None

        # Parameter mapping from pytorch to keras
        torch_paramap = {
            # Conv
            'kernel': 'weight',
            # Batchnorm
            'gamma': 'weight',
            'beta': 'bias',
            'moving_mean': 'running_mean',
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