import torch.nn as nn

from collections import OrderedDict
from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)

import numpy as np

def conv_block(in_channels, out_channels, pool_size=2, dropout=0, **kwargs):
    return MetaSequential(OrderedDict([
        ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
        ('norm', nn.BatchNorm2d(out_channels, momentum=1., track_running_stats=False)),
        ('relu', nn.Softplus()),
        ('pool', nn.MaxPool2d(pool_size)),
        ('drop', nn.Dropout(dropout))
    ]))


class MetaConvModel(MetaModule):
    """Modified to be able to have variable structures
    instead of just the 4-layer Convolutional Neural Network architecture from [1].

    Parameters
    ----------
    in_channels : int
        Number of channels for the input images. Used in architeture from [1]

    out_features : int
        Number of classes (output of the model).
        
    params_file : str
        Path to file that contains the network parameters (structure, lr, etc...)
        
    use_original : bool
        Bool indicating whether or not the original architecture from [1] should be used for comparison purposes
        
    with_output_layer : bool
        parameter from meta_lenet_decolle.py and used in the same manner
        
    remove_time_dim : bool
        Indicates whether or not to use the time dimension, which would make it closer to an SNN 

    hidden_size : int (default: 64)
        Number of channels in the intermediate representations. Used in architeture from [1]

    feature_size : int (default: 64)
        Number of features returned by the convolutional head. Used in architeture from [1]

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """

    def __init__(self, in_channels, out_features, params_file=None, use_original=False, with_output_layer=True,  remove_time_dim=False, hidden_size=64, feature_size=64):
        super(MetaConvModel, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size #Used only if original
        self.feature_size = feature_size #Used only if original
        self.remove_time_dim = remove_time_dim
        
        if not use_original:
        
            with open(params_file, 'r') as f:
                import yaml
                params = yaml.load(f)
            verbose = True

            self.Nhid = params['Nhid']
            self.Mhid = params['Mhid']
            self.stride = params['stride']
            self.num_conv_layers = params['num_conv_layers']
            self.num_mlp_layers = params['num_mlp_layers']
            self.kernel_size = params['kernel_size']
            self.pool_size = params['pool_size']
            self.input_shape = params['input_shape']

            dropout=[0] #[0.5]

            if with_output_layer:
                self.Mhid += [out_features]
                self.num_mlp_layers+=1

            self.num_layers = num_layers = self.num_conv_layers + self.num_mlp_layers
            if len(self.kernel_size) == 1:   self.kernel_size = self.kernel_size * self.num_conv_layers
            if self.stride is None: self.stride=[1]
            if len(self.stride) == 1:        self.stride = self.stride * self.num_conv_layers
            if self.pool_size is None: self.pool_size = [1]
            if self.Nhid is None:          self.Nhid = Nhid = []
            if self.Mhid is None:          self.Mhid = Mhid = []

            if len(dropout) == 1:       self.dropout = dropout = dropout * num_layers

            self.padding = (np.array(self.kernel_size) - 1) // 2

            self.Nhid = [self.input_shape[0]] + self.Nhid

            feature_height = self.input_shape[1]
            feature_width = self.input_shape[2]

            layers = OrderedDict()
            for i in range(self.num_conv_layers):
                layers[f'layer{i}'] = conv_block(
                        self.Nhid[i],
                        self.Nhid[i + 1], 
                        self.pool_size[i], 
                        self.dropout[i], 
                        kernel_size=self.kernel_size[i], 
                        stride=self.stride[i], 
                        padding=self.padding[i],
                        bias=True)
                feature_height, feature_width = get_output_shape(
                    [feature_height, feature_width], 
                    kernel_size = self.kernel_size[i],
                    stride = self.stride[i],
                    padding = self.padding[i],
                    dilation = 1)
                feature_height //= self.pool_size[i]
                feature_width //= self.pool_size[i]

            self.features = MetaSequential(layers)
            conv_stack_output_shape = self.Nhid[i+1],feature_height,feature_width
            mlp_in = int(np.prod(conv_stack_output_shape))
            self.Mhid = [mlp_in] + self.Mhid

            self.classifier = MetaLinear(self.Mhid[0], self.Mhid[1],  bias=True)
        
        if use_original:
            in_channels = self.in_channels
            hidden_size = self.hidden_size
            feature_size = self.feature_size
            
            self.features = MetaSequential(OrderedDict([
                ('layer1', conv_block(in_channels, hidden_size, kernel_size=3,
                                      stride=1, padding=1, bias=True)),
                ('layer2', conv_block(hidden_size, hidden_size, kernel_size=3,
                                      stride=1, padding=1, bias=True)),
                ('layer3', conv_block(hidden_size, hidden_size, kernel_size=3,
                                      stride=1, padding=1, bias=True)),
                ('layer4', conv_block(hidden_size, hidden_size, kernel_size=3,
                                      stride=1, padding=1, bias=True))
            ]))
            self.classifier = MetaLinear(feature_size, out_features, bias=True)
        
        
    def forward(self, inputs, params=None):
        if self.remove_time_dim: inputs = inputs[:,0]
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits
    
    def get_trainable_parameters(self):
        params = list()
        for p in self.parameters():
            if p.requires_grad:           
                params.append(p)
        return params
    
    def get_trainable_named_parameters(self):
        params = dict()
        for k,p in self.named_parameters():
            if p.requires_grad:
                params[k]=p

        return params

    
    
class MetaMLPModel(MetaModule):
    """Multi-layer Perceptron architecture from [1].

    Parameters
    ----------
    in_features : int
        Number of input features.

    out_features : int
        Number of classes (output of the model).

    hidden_sizes : list of int
        Size of the intermediate representations. The length of this list
        corresponds to the number of hidden layers.
        
    flatten_spatial : bool
        Useful for flattening image data for MLP

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. International Conference on Machine Learning (ICML) (https://arxiv.org/abs/1703.03400) Modified here to use Tanh instead of ReLU
    """
    def __init__(self, in_features, out_features, hidden_sizes, flatten_spatial=False):
        super(MetaMLPModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes
        self.flatten_spatial = flatten_spatial

        layer_sizes = [in_features] + hidden_sizes
        if self.flatten_spatial:
            self.flatten = nn.Flatten()
        self.features = MetaSequential(OrderedDict([('layer{0}'.format(i + 1),
            MetaSequential(OrderedDict([
                ('linear', MetaLinear(hidden_size, layer_sizes[i + 1], bias=True)),
                ('relu', nn.Tanh())
            ]))) for (i, hidden_size) in enumerate(layer_sizes[:-1])]))
        self.classifier = MetaLinear(hidden_sizes[-1], out_features, bias=True)

    def forward(self, inputs, params=None):
        if self.flatten_spatial:
            inputs = self.flatten(inputs)
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits
    
    def get_trainable_parameters(self):
        params = list()
        for p in self.parameters():
            if p.requires_grad:           
                params.append(p)
        return params
    
    def get_trainable_named_parameters(self):
        params = dict()
        for k,p in self.named_parameters():
            if p.requires_grad:
                params[k]=p

        return params

def ModelConvOmniglot(out_features, hidden_size=64):
    return MetaConvModel(1, out_features, hidden_size=hidden_size,
                         feature_size=hidden_size, use_original=True)

def ModelConvDoubleNMNIST(out_features, params_file, use_original=False, remove_time_dim=True): #hidden_size=64):
    if remove_time_dim:
        return MetaConvModel(2, out_features, params_file=params_file, use_original=use_original, remove_time_dim=remove_time_dim) 
    else:
        return TimeWrappedMetaConvModel(2, out_features, params_file=params_file, use_original=use_original, remove_time_dim=remove_time_dim) 
    
    #hidden_size=hidden_size, feature_size=256, remove_time_dim=True)



def ModelConvMiniImagenet(out_features, hidden_size=64):
    return MetaConvModel(3, out_features, hidden_size=hidden_size,
                         feature_size=5 * 5 * hidden_size)

def ModelMLPSinusoid(hidden_sizes=[40, 40]):
    return MetaMLPModel(1, 1, hidden_sizes)

def ModelMLPOmniglot(out_features, hidden_size):
    return MetaMLPModel(784, out_features, hidden_sizes = [hidden_size]*3, flatten_spatial=True)

