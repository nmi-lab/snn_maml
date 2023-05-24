#!/bin/python
#-----------------------------------------------------------------------------
# File Name : meta_lenet_decolle.py
# Author: Emre Neftci
#
# Creation Date : Tue 08 Sep 2020 11:18:03 AM PDT
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 

from decolle.base_model import *
from decolle.lenet_decolle_model import *
from collections import OrderedDict
from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)

import torch.nn as nn

from decolle.utils import get_output_shape

import lava.lib.dl.slayer as slayer
from lava.lib.dl.slayer.metamodule import MetaModuleNg

import numpy as np

import warnings

m = nn.Sigmoid()

class FastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        return  input_ / (1+torch.abs(input_))

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input / (torch.abs(input_) + 1.0) ** 2
    
fast_sigmoid = FastSigmoid.apply



class MetaModuleNg(MetaModule):
    """
    MetaModule that returns only elements that require_grad
    """
    def meta_named_parameters(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items() if isinstance(module, MetaModule) else [],
            prefix=prefix, recurse=recurse)
        for elem in gen:
            if elem[1].requires_grad:
                yield elem
                
class MetaLenetDECOLLE(LenetDECOLLE,MetaModuleNg):
    def __init__(self, burnin, detach_at=-1, sg_function_baseline = False, *args, **kwargs):
        self.non_spiking_baseline = sg_function_baseline
        if self.non_spiking_baseline is True:
            print('Using non-spiking model!')
        super(MetaLenetDECOLLE, self).__init__(*args, **kwargs)
        self.burnin = burnin
        self.detach_at = detach_at
    
    def build_conv_stack(self, Nhid, feature_height, feature_width, pool_size, kernel_size, stride, out_channels):
        output_shape = None
        padding = (np.array(kernel_size) - 1) // 2  
        for i in range(self.num_conv_layers):
            feature_height, feature_width = get_output_shape(
                [feature_height, feature_width], 
                kernel_size = kernel_size[i],
                stride = stride[i],
                padding = padding[i],
                dilation = 1)
            feature_height //= pool_size[i]
            feature_width //= pool_size[i]
            base_layer = MetaConv2d(Nhid[i], Nhid[i + 1], kernel_size[i], stride[i], padding[i])
            layer = self.lif_layer_type[i](base_layer,
                             alpha=self.alpha[i],
                             beta=self.beta[i],
                             alpharp=self.alpharp[i],
                             wrp=self.wrp[i],
                             deltat=self.deltat,
                             do_detach= True if self.method == 'rtrl' else False)
            pool = nn.AvgPool2d(kernel_size=pool_size[i])
            if self.lc_ampl is not None:
                readout = nn.Linear(int(feature_height * feature_width * Nhid[i + 1]), out_channels)

                # Readout layer has random fixed weights
                for param in readout.parameters():
                    param.requires_grad = False
                self.reset_lc_parameters(readout, self.lc_ampl[i])
            else:
                readout = nn.Identity()
            self.readout_layers.append(readout)

            if self.dropout[i] > 0.0:
                dropout_layer = nn.Dropout(self.dropout[i])
            else:
                dropout_layer = nn.Identity()


            self.LIF_layers.append(layer)
            self.pool_layers.append(pool)
            self.dropout_layers.append(dropout_layer)
        return (Nhid[-1],feature_height, feature_width)

    def build_mlp_stack(self, Mhid, out_channels): 
        output_shape = None

        for i in range(self.num_mlp_layers):
            base_layer = MetaLinear(Mhid[i], Mhid[i+1])
            layer = self.lif_layer_type[i+self.num_conv_layers](base_layer,
                         alpha=self.alpha[i],
                         beta=self.beta[i],
                         alpharp=self.alpharp[i],
                         wrp=self.wrp[i],
                         deltat=self.deltat,
                         do_detach=True if self.method == 'rtrl' else False)
            if self.lc_ampl is not None:
                readout = nn.Linear(Mhid[i+1], out_channels)
                # Readout layer has random fixed weights
                for param in readout.parameters():
                    param.requires_grad = False
                self.reset_lc_parameters(readout, self.lc_ampl[i])
            else:
                readout = nn.Identity()

            if self.dropout[i] > 0.0:
                dropout_layer = nn.Dropout(self.dropout[i])
            else:
                dropout_layer = nn.Identity()
            output_shape = out_channels

            self.LIF_layers.append(layer)
            self.pool_layers.append(nn.Sequential())
            self.readout_layers.append(readout)
            self.dropout_layers.append(dropout_layer)
        return (output_shape,)

    def build_output_layer(self, Mhid, out_channels):
        if self.with_output_layer:
            i=self.num_mlp_layers
            base_layer = MetaLinear(Mhid[i], out_channels)
            layer = self.lif_layer_type[-1](base_layer,
                         alpha=self.alpha[i],
                         beta=self.beta[i],
                         alpharp=self.alpharp[i],
                         wrp=self.wrp[i],
                         deltat=self.deltat,
                         do_detach=True if self.method == 'rtrl' else False)
            readout = nn.Identity()
            if self.dropout[i] > 0.0:
                dropout_layer = nn.Dropout(self.dropout[i])
            else:
                dropout_layer = nn.Identity()
                
            output_shape = out_channels

            self.LIF_layers.append(layer)
            self.pool_layers.append(nn.Sequential())
            self.readout_layers.append(readout)
            self.dropout_layers.append(dropout_layer)
            
        return (output_shape,)

    def step(self, input, params = None):
        s_out = []
        r_out = []
        u_out = []
        i = 0
        for lif, pool, ro, do in zip(self.LIF_layers, self.pool_layers, self.readout_layers, self.dropout_layers):
            if i == self.num_conv_layers: 
                input = input.view(input.size(0), -1)
            s, u = lif(input, self.get_subdict(params,'LIF_layers.{0}.base_layer'.format(i)))
            if i==self.detach_at:
                warnings.warn('detaching layer {0}'.format(lif))
                s=s.detach()
                u=u.detach()
            u_p = pool(u)
            if i+1 == self.num_layers and self.with_output_layer:
                s_ = sigmoid(u_p)
                #sd_ = u_p
                #r_ = ro(sd_.reshape(sd_.size(0), -1))
            elif self.non_spiking_baseline:
                s_ = fast_sigmoid(u_p) #m(u_p) #Fastsigmoid
            else:
                s_ = lif.sg_function(u_p)
                #sd_ = do(s_)
                #r_ = ro(sd_.reshape(sd_.size(0), -1))

            s_out.append(s_) 
            #r_out.append(r_)
            u_out.append(u_p)
            input = s_.detach() if lif.do_detach else s_
            i+=1

        return s_out, r_out, u_out

class MetaRecLIFLayer(LIFLayer,MetaModuleNg):
    def __init__(*args, **kwargs):
        raise NotImplementedError('See previous commits of pytorch-maml:update__kennetms')

class MetaLIFLayer(LIFLayer,MetaModuleNg):
    def forward(self, Sin_t, params = None, *args, **kwargs ): 
        if self.state is None:
            self.init_state(Sin_t)
        if Sin_t.shape[0] != self.state.P.shape[0]:
            warnings.warn('Reinitializing state')
            self.init_state(Sin_t)
        
        state = self.state
        Q = self.beta * state.Q + (1-self.beta)*Sin_t*self.gain
        P = self.alpha * state.P + (1-self.alpha)*state.Q
        R = self.alpharp * state.R - (1-self.alpharp)*state.S * self.wrp
        U = self.base_layer(P, params=params) + R
        S = self.sg_function(U)
        self.state = self.NeuronState(P=P, Q=Q, R=R, S=S)
        if self.do_detach: 
            state_detach(self.state)
        return S, U
    
    def init_parameters(self, *args, **kwargs):
        self.reset_parameters(self.base_layer, *args, **kwargs)
        

        
class SLAYERConvNetwork(MetaModuleNg):
    neuron_params = {
            'threshold': 1.0,
            'current_decay': 0.10,
            'voltage_decay': 0.03,
            'tau_grad': 0.1,
            'scale_grad': 5,
            'requires_grad': False,
        }
    neuron_params_drop   = {**neuron_params, 'dropout': slayer.neuron.Dropout(p=0.0), }
    neuron_params_final = {
        'threshold': 1.0,
        'current_decay': 0.10,
        'voltage_decay': 0.03,
        'tau_grad': 0.1,
        'scale_grad': 5,
        'requires_grad': False,
        'debug':True
    }
    def __init__(self, output_features, width=28, height=28 , analog_readout=True):
        super(SLAYERConvNetwork, self).__init__()
        self.build_network(output_features, width, height)
        self.analog_readout = analog_readout
        if analog_readout:
            self.blocks[-1].neuron.return_internal_state = True
            


    def build_network(self, output_features, width, height):


        
        self.blocks = torch.nn.ModuleList([
            # slayer.block.cuba.Input(neuron_params=self.neuron_params_drop),
#            slayer.block.cuba.Pool(neuron_params=neuron_params_drop,
#                                   kernel_size=2,
#                                   delay=False,
#                                   delay_shift=False,
#                                   stride=2),
            slayer.block.cuba.Conv(neuron_params=self.neuron_params_drop,
                                   in_features=2,
                                   out_features=16,
                                   kernel_size=5,
                                   delay=False,
                                   delay_shift=False,
                                   padding=2,
                                   weight_scale=1,
                                   weight_norm=False),
            
            slayer.block.cuba.Pool(neuron_params=self.neuron_params,
                                   kernel_size=2,
                                   stride=2),
            
            slayer.block.cuba.Conv(neuron_params=self.neuron_params_drop,
                                   in_features=16,
                                   out_features=32,
                                   kernel_size=3,
                                   padding=1, 
                                   delay=False,
                                   delay_shift=False,
                                   weight_scale=1,
                                   weight_norm=False),
            
            slayer.block.cuba.Pool(neuron_params=self.neuron_params,
                                   kernel_size=2,
                                   stride=2),
            
            slayer.block.cuba.Flatten(),
            
            slayer.block.cuba.Dense(neuron_params=self.neuron_params_drop,
                                    in_neurons=1024, #hardcoded for nomniglot
                                    out_neurons=512,
                                    delay=False,
                                    delay_shift=False,
                                    weight_scale=1,
                                    weight_norm=False),
            
            slayer.block.cuba.Dense(neuron_params=self.neuron_params_final,
                                    in_neurons=512,
                                    out_neurons=output_features,
                                    delay=False,
                                    delay_shift=False,
                                    weight_scale=1,
                                    weight_norm=False,
                                    pre_hook_fx=None),
        ])

    def forward(self, spike, params=None):
        # print(spike.size())
        count = []
        spike = spike.permute(0,2,3,4,1)
        
        if params is None:
            params = OrderedDict(self.named_parameters())
            
        count = []
        
        for block_id, block in enumerate(self.blocks):
            spike = block(spike, 
                          params = self.get_subdict(params,
                                                        'blocks.{0}'.format(block_id)))        
        # if spike[0].shape[0]==50:
            # raise
        if self.analog_readout:
            return spike[1][:,:,:]
        else:
            return torch.mean(spike[:,:,50:], axis=2)

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [
            b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')
        ]

        import matplotlib.pyplot as plt
        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

    def export_hdf5(self, filename, add_input_layer=False, input_dims=[2, 128, 128]):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        if add_input_layer:
            input_layer = layer.create_group(f'{0}')
            input_layer.create_dataset('shape', data=np.array(input_dims))
            input_layer.create_dataset('type', (1, ), 'S10', ['input'.encode('ascii', 'ignore')])
        for i, b in enumerate(self.blocks):
            if add_input_layer:
                b.export_hdf5(layer.create_group(f'{i+1}'))
            else:
                b.export_hdf5(layer.create_group(f'{i}'))
        # add simulation key for nxsdk
        sim = h.create_group('simulation')
        sim.create_dataset('Ts', data=1)
        sim.create_dataset('tSample', data=1500)

    def gen_loihi_params(self, folder):
        # Define Loihi parameter generator
        for i, b in enumerate(self.blocks):
            if hasattr(b, 'synapse'):
                if hasattr(b.synapse, 'weight'):
                    print(b.synapse.weight.cpu().data.numpy())
                    weights = slayer.utils.quantize(b.synapse.weight, step=2).cpu().data.numpy()
                    weights = np.squeeze(weights, axis=-1)
                    print('Block: ' + str(b) + ' layer: ' + str(i) + ' with shape:' + str(np.shape(weights)))
                    print(np.max(weights))
                    print(weights)
                    np.save(folder + '/' + f'{i}' + '_weights.npy', weights)
                if b.delay:
                    delays = slayer.utils.quantize(b.delay.delay, step=2).cpu().data.numpy()
                    # delays = np.squeeze(delays, axis=-1)
                    np.save(folder + '/' + f'{i}' + '_delays.npy', delays)
                    print('Delay layer: ' + str(i) + ' with shape:' + str(np.shape(delays)))
                    
class SLAYERDenseNetwork(SLAYERConvNetwork):
    def build_network(self, output_features, width, height):
        #neuron_params_drop = {**neuron_params}

        
        self.blocks = torch.nn.ModuleList([
            #slayer.block.cuba.Input(neuron_params=neuron_params_drop),
#            slayer.block.cuba.Pool(neuron_params=neuron_params_drop,
#                                   kernel_size=2,
#                                   delay=False,
#                                   delay_shift=False,
#                                   stride=2),
           
            slayer.block.cuba.Flatten(),
            
            slayer.block.cuba.Dense(neuron_params=self.neuron_params_drop,
                                    in_neurons=width*height*2, #hardcoded for nomniglot
                                    out_neurons=512,
                                    delay=True,
                                    delay_shift=False,
                                    weight_scale=1,
                                    weight_norm=True),
            
            slayer.block.cuba.Dense(neuron_params=self.neuron_params_final,
                                    in_neurons=512,
                                    out_neurons=output_features,
                                    delay=False,
                                    delay_shift=False,
                                    weight_scale=1,
                                    weight_norm=False,
                                    pre_hook_fx=None),
        ])
        #self.blocks[-1].neuron.return_internal_state = True

    


def build_model_DECOLLE(out_features, params_file, device, detach_at, sg_function_baseline): 
    # = 'maml/decolle_params-CNN.yml'
    from decolle.utils import parse_args, prepare_experiment, cross_entropy_one_hot
    import datetime, os, socket, tqdm
    import torch
    import torch.nn.functional as F

    params_file = params_file 
    with open(params_file, 'r') as f:
        import yaml
        params = yaml.safe_load(f)
    verbose = True
    
    reg_l = params['reg_l'] if 'reg_l' in params else None

    #d, t = next(iter(gen_train))
    input_shape = params['input_shape']
    ## Create Model, Optimizer and Loss
    net = MetaLenetDECOLLE(
                        out_channels=out_features,
                        Nhid=params['Nhid'],
                        Mhid=params['Mhid'],
                        kernel_size=params['kernel_size'],
                        pool_size=params['pool_size'],
                        stride = params['stride'],
                        input_shape=params['input_shape'],
                        alpha=params['alpha'],
                        alpharp=params['alpharp'],
                        beta=params['beta'],
                        dropout=params['dropout'],
                        num_conv_layers=params['num_conv_layers'],
                        num_mlp_layers=params['num_mlp_layers'],
                        lc_ampl=params['lc_ampl'],
                        lif_layer_type = MetaLIFLayer,
                        method=params['learning_method'],
                        with_output_layer=True,
                        wrp=params['wrp'],
                        burnin=params['burnin_steps'],
                        detach_at=detach_at,
                        sg_function_baseline=sg_function_baseline).to(device)
    
    net.LIF_layers[0].gain=10
    
    ##Initialize
    net.init_parameters(torch.zeros([1,params['chunk_size_train']]+params['input_shape']).to(device))

    return net

def build_model_SLAYER(out_features, params_file, device, network_type='conv'): # = 'maml/decolle_params-CNN.yml'
    from decolle.utils import parse_args, prepare_experiment, cross_entropy_one_hot
    import datetime, os, socket, tqdm
    import torch
    import torch.nn.functional as F

    params_file = params_file 
    with open(params_file, 'r') as f:
        import yaml
        params = yaml.safe_load(f)
    verbose = True
    
    input_shape = params['input_shape']
    ## Create Model, Optimizer and Loss

    if network_type == 'conv':
        net = SLAYERConvNetwork(out_features, input_shape[1], input_shape[2]).to(device)
        net.parameters_to_train = ['blocks.6.synapse.weight']
    elif network_type == 'dense':
        net = SLAYERDenseNetwork(out_features, input_shape[1], input_shape[2]).to(device)
        net.parameters_to_train = ['blocks.2.synapse.weight']
                                   
    else:
        raise RuntimeError('network type must be conv or dense')
        
    

    return net


def build_model_REDECOLLE(out_features, params_file, rec_layer=MetaRecLIFLayer, out_layer=MetaLIFLayer):
    from decolle.utils import parse_args, prepare_experiment, cross_entropy_one_hot
    import datetime, os, socket, tqdm
    import torch
    import torch.nn.functional as F

    params_file = params_file 
    with open(params_file, 'r') as f:
        import yaml
        params = yaml.load(f)
    verbose = True
    
    reg_l = params['reg_l'] if 'reg_l' in params else None

    #d, t = next(iter(gen_train))
    input_shape = params['input_shape']
    ## Create Model, Optimizer and Loss
    rec_lif_layer_type = rec_layer


    net = MetaLenetREDECOLLE(
                        out_channels=out_features,
                        Nhid=params['Nhid'],
                        Mhid=params['Mhid'],
                        kernel_size=params['kernel_size'],
                        pool_size=params['pool_size'],
                        stride = params['stride'],
                        input_shape=params['input_shape'],
                        alpha=params['alpha'],
                        alpharp=params['alpharp'],
                        beta=params['beta'],
                        dropout=params['dropout'],
                        num_conv_layers=params['num_conv_layers'],
                        num_mlp_layers=params['num_mlp_layers'],
                        lc_ampl=params['lc_ampl'],
                        lif_layer_type = [MetaLIFLayer]*len(params['Nhid'])+[rec_layer]*len(params['Mhid'])+[out_layer],
                        method=params['learning_method'],
                        with_output_layer=True,
                        wrp=params['wrp'],
                        burnin=params['burnin_steps']).cuda()
    
    net.init_parameters(torch.zeros([1,params['chunk_size_train']]+params['input_shape']).cuda())

    return net

