#!/bin/python
#-----------------------------------------------------------------------------
# File Name : snn_model_lava.py
# Author: Kenneth Stewart
#
# Creation Date : Tue 20 Sep 2022 11:18:03 AM PDT
# Last Modified : 
#
# Copyright : (c) Kenneth Stewart, Michael Neumeier
# Licence : GPLv2
#----------------------------------------------------------------------------- 

from collections import OrderedDict
from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)

import lava.lib.dl.slayer as slayer

from lava.lib.dl.slayer.synapse.layer import *

import torch.nn as nn

from decolle.utils import get_output_shape

import numpy as np

import torch

import warnings

import pdb

from matplotlib import pyplot as plt

import random

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
                
                
def get_voltage_current_slayer_network(spike, model, end_block_id):
    
    spike = spike.permute(0,2,3,4,1)
    
    #spike = spike.flatten(1,3)
    
    prev_value = model.blocks[end_block_id].neuron.return_internal_state
    model.blocks[end_block_id].neuron.return_internal_state=True

    # model should have delay, delay_shift and weight_norm=False
    for block_id, block in enumerate(model.blocks[:end_block_id+1]):
        if isinstance(block.synapse, MetaDense) and len(spike.shape)>3:
            spike = spike.flatten(1,3)
        spike, volt = block(spike)     

    model.blocks[end_block_id].neuron.return_internal_state=prev_value
    return (spike, volt)
    
def torch_init_LSUV(model, data_batch, block_ids, tgt_mu=0.0, tgt_var=1.0):
    '''
    Initialization inspired from Mishkin D and Matas J. All you need is a good init. arXiv:1511.06422 [cs],
February 2016.
    '''
    import torch
    from torch.nn import init

    device = model.blocks[block_ids[0]].neuron.device
    ##Initialize
    with torch.no_grad():
        #def lsuv(model, data_batch):
        for block_id in block_ids:
            init.orthogonal_(model.blocks[block_id].synapse.weight.data)

        for block_id in block_ids:
            count=0
            alldone = False
            while not alldone:
                alldone = True
                state_layer = get_voltage_current_slayer_network(data_batch, model, block_id)
                synapse = model.blocks[block_id].synapse
                weight = model.blocks[block_id].synapse.weight
                current = state_layer[1].cpu()
                spikes = state_layer[0].cpu()
                #pdb.set_trace()
                v = np.var(current.flatten().numpy())
                m = np.mean(current.flatten().numpy())
                mus=np.mean(spikes.flatten().numpy()) 
                print("Layer: {0}, Variance: {1:.3}, Mean U: {2:.3}, Mean S: {3:.3}".format(block_id, v, m, mus))
                if np.isnan(v) or np.isnan(m):
                    print('Nan encountered during init')
                    done = False
                    raise
                if np.abs(v - tgt_var) > .2:#.2:
                    if block_id >= 0:#0:
                        # smaller steps for deeper layers
                        weight.data /= (0.8 + .2 * np.sqrt(v))#(0.80 + .2 * np.sqrt(v))
                        weight.data *= (0.8 + .2 * np.sqrt(tgt_var))#(0.80 + .2 * np.sqrt(tgt_var))
                    else:
                        weight.data *= np.sqrt(tgt_var) / (np.sqrt(v) + 1e-8)
                    done = False
                else:
                    done = True
                alldone *= done

                if np.abs(m - tgt_mu) > .3:#.3:
                    weight.data -= torch.tensor(.0001 * (m - tgt_mu), dtype=weight.dtype)#.0001
                    done = False
                else:
                    done = True
                alldone *= done
                
                count+=1
                if count >= 100:
                    alldone=True

            if alldone:
                print("Initialization finalized:")
                print("Layer: {0}, Variance: {1:.3}, Mean U: {2:.3}".format(block_id, v, m))
                print("-------------------------------------")
                
def init_LSUV_actrate(net, data_batch, act_rate, threshold=0., var=1.0):
    from scipy.stats import norm
    import scipy.optimize
    tgt_mu = scipy.optimize.fmin(lambda loc: (act_rate-(1-norm.cdf(threshold,loc,var)))**2, x0=0.)[0]
    init_LSUV(net, data_batch, tgt_mu=tgt_mu, tgt_var=var)
                
        
class LavaNet(nn.Module):
    def __init__(self, params):
        super(LavaNet, self).__init__()
        
        self.network_params = params['network']
        
        self.neuron_params = params['neuron_model']
        
        self.neuron_params_dropout = {**params['neuron_model'], 'dropout' : slayer.neuron.Dropout(p=0.0)} # was .05. from looking at the example input and hidden dense layers used this. Not sure about conv, would have to experiment I guess
        
        #pdb.set_trace()
        
        self.blocks = MetaSequential()
        
        self.input_given = False
        
        self.Mhid = []
        
        if self.network_params['Nhid']:
            conv_stack_output_shape = self.build_conv_stack()
            mlp_in = int(np.prod(conv_stack_output_shape))
            
            self.Mhid = [mlp_in] + self.network_params['Mhid']
            self.input_given = True

        if self.network_params['Mhid']:
            mlp_stack_output_shape = self.build_mlp_stack()
            
            if not self.Mhid:
                self.Mhid = [mlp_stack_output_shape]

        if self.network_params['out_channels'] is not None:
            output_shape = self.build_output_layer()
            
        self.analog_readout=self.network_params['analog_readout'] # Emre defaults to True
        
        if self.analog_readout:
            self.blocks[-1].neuron.return_internal_state = True
            
        
        for block in self.blocks:
            block.neuron.quantize = self.neuron_params['quantize']
            
            
        # use data initialization
        print(self.blocks)
        
        
    def build_conv_stack(self):    
        # slayer.block.sigma_delta.Conv(sdnn_cnn_params,  in, out, kernel, padding=0, stride=2, weight_scale=2, weight_norm=True) just an example for reference
        i = 0
        
        feature_height = self.network_params['input_shape'][1]
        feature_width = self.network_params['input_shape'][2]
        
        for nhid in range(len(self.network_params['Nhid'])):
            
            padding = (self.network_params['kernel_size'][nhid] - 1) // 2
            
            if not self.input_given:
                self.blocks.append(slayer.block.cuba.Conv(self.neuron_params, self.network_params['input_shape'][0], self.network_params['Nhid'][nhid], kernel_size=self.network_params['kernel_size'][nhid], stride=self.network_params['stride'][nhid],padding=padding, weight_scale=1, weight_norm=False,delay=self.network_params['delay'],delay_shift=True,pre_hook_fx=None if self.neuron_params['quantize'] else lambda x: x))
                self.input_given = True
                # pool params: kernel_size, stride=None, padding=0, dilation=1, weight_scale=1, weight_norm=False, pre_hook_fx=None), remember pre_hook_fx is for quantization so I'll need to add this for loihi
                self.blocks.append(slayer.block.cuba.Pool(self.neuron_params,kernel_size=self.network_params['pool_size'][nhid],delay=self.network_params['delay'],delay_shift=False,pre_hook_fx=None if self.neuron_params['quantize'] else lambda x: x))
                
            else:
                self.blocks.append(slayer.block.cuba.Conv(self.neuron_params, self.network_params['Nhid'][nhid-1], self.network_params['Nhid'][nhid], kernel_size=self.network_params['kernel_size'][nhid], stride=self.network_params['stride'][nhid],padding=padding,weight_scale=1, weight_norm=False,delay=self.network_params['delay'],delay_shift=True,pre_hook_fx=None if self.neuron_params['quantize'] else lambda x: x))
                self.blocks.append(slayer.block.cuba.Pool(self.neuron_params,kernel_size=self.network_params['pool_size'][nhid],delay=self.network_params['delay'],delay_shift=False,pre_hook_fx=None if self.neuron_params['quantize'] else lambda x: x))
                
            
            feature_height, feature_width = self.get_output_shape(
                [feature_height, feature_width], 
                kernel_size = self.network_params['kernel_size'][nhid],
                stride = self.network_params['stride'][nhid],
                padding = (self.network_params['kernel_size'][nhid] - 1) // 2, #[1] if self.network_params['padding'][nhid] is None else [self.network_params['padding'][nhid]],
                dilation = 1)
            
            feature_height //= self.network_params['pool_size'][nhid]
            feature_width //= self.network_params['pool_size'][nhid]
                
        return (self.network_params['Nhid'][-1],feature_height, feature_width)
                                   
    def build_mlp_stack(self):
        # slayer.block.cuba.Dense(neuron_params, 34*34*2, 512, weight_norm=True, delay=True) example for reference

        for mhid in range(len(self.network_params['Mhid'])):
            if not self.input_given:
                self.blocks.append(slayer.block.cuba.Dense(self.neuron_params_dropout, int(np.prod(self.network_params['input_shape'])), self.network_params['Mhid'][mhid], weight_norm=False, delay=self.network_params['delay'],delay_shift=True,weight_scale=1,pre_hook_fx= None if self.neuron_params['quantize'] else lambda x: x))#,pre_hook_fx=lambda x: x))#pre_hook_fx=lambda x: x))
                self.input_given = True
            elif self.Mhid:
                self.blocks.append(slayer.block.cuba.Dense(self.neuron_params_dropout, self.Mhid[mhid], self.Mhid[mhid+1], weight_norm=False, delay=self.network_params['delay'],delay_shift=True,weight_scale=1, pre_hook_fx=None if self.neuron_params['quantize'] else lambda x: x))#,pre_hook_fx=lambda x: x))#pre_hook_fx=lambda x: x))
            else:
                 self.blocks.append(slayer.block.cuba.Dense(self.neuron_params_dropout, self.network_params['Mhid'][mhid-1], self.network_params['Mhid'][mhid], weight_norm=False, delay=self.network_params['delay'],delay_shift=True,weight_scale=1, pre_hook_fx=None if self.neuron_params['quantize'] else lambda x: x))#,pre_hook_fx=lambda x: x))#pre_hook_fx=lambda x: x))
                    
        return (self.network_params['Mhid'][-1])

    # can try to set weight_norm to True, should act like batch norm which could help
    # try adding refractory dynamics, alif in lava has it (lif with adaptive threshold, could turn off for lif)
    # examine spike activity of the layers, average spike rate in the layers, ~10-50% ballpark
    # if spiking too much, lower initial weight values. Too little then raise. weight_scale param
    # make sure layers keep spiking and are not dying
    # make sure graidents of inner and outer loop are not vanishing or exploding
    
    def build_output_layer(self):
        self.blocks.append(slayer.block.cuba.Dense(self.neuron_params, self.Mhid[-1], self.network_params['out_channels'], weight_norm=False,delay=self.network_params['delay'],delay_shift=self.network_params['delay'], weight_scale=1, pre_hook_fx=None if self.neuron_params['quantize'] else lambda x: x))#,pre_hook_fx=lambda x: x))#,pre_hook_fx=None))
        return (self.network_params['out_channels']) #bias not compatible with loihi 1, but is with loihi2 with microcode. Bias can work on loihi 1 if the current decay is 4096, then it will work
    
    def get_output_shape(self, input_shape, kernel_size, stride, padding, dilation):
        #pdb.set_trace()
        # must be incorrect since dvssign doesn't work with cnn
        im_height = input_shape[-2]
        im_width  = input_shape[-1]
        
        height = int((im_height + 2 * padding - dilation *
                          (kernel_size - 1) - 1) // stride + 1)
        # if len(kernel_size)>1:
        #     width = int((im_width + 2 * padding - dilation *
        #                   (kernel_size - 1) - 1) // stride + 1)
        # else:
        width = int((im_width + 2 * padding - dilation *
                          (kernel_size - 1) - 1) // stride + 1)
            
        return [height, width]
    
    def forward(self, spike):
        #self.analog_readout=False
        #self.blocks[-1].neuron.return_internal_state = False
        #spike = self.transpose_torchneuromorphic_to_SLAYER(spike)
        
        spike_input = spike.permute(0,2,3,4,1)
        
        if not self.network_params['Nhid']:
            # first layer is mlp, flatten input
            spike = spike.flatten(1,3)
            # try something like burnin which skips a certain number of steps
            #spike = spike[:,:,70:]
            
        params_wb = OrderedDict()
            
        for k in range(1): # really simple way of doing a burnin type thing
            spike = spike_input
            i = 0
            j = 0
            for block in self.blocks:
                if self.network_params['Nhid']:
                    if i%2==0 and j!=(len(self.network_params['Nhid'])*2):
                        spike, volt = block(spike)
                        j+=1
                    elif j>=(len(self.network_params['Nhid'])*2):
                        # should be linear or output layer, need to flatten input
                        if len(spike.shape)>3:
                            spike = spike.flatten(1,3)
                        spike, volt = block(spike)
                    else:
                        # should be pooling layer
                        spike, volt = block(spike)
                        j+=1

                else:

                    if len(spike.shape)>3:
                        spike = spike.flatten(1,3)
                    
                    #pdb.set_trace()
                    spike, volt = block(spike)
                    #print("before final layer")
                    #pdb.set_trace

                i+=1

        # in Emre's implementation he has this, which could be an important difference for how the performance is
        # his is also using torch's cross entropy loss instead of a slayer loss, which could also be a difference maker
        #pdb.set_trace()
        if self.analog_readout:
            #pdb.set_trace()
            #print("analog readout")
            #return torch.mean(curr_volt[1][:,:],axis=1)
            # spike
            return spike, volt#torch.sum(volt[:,:],axis=2)#,-1]#-1 #[1][:,:]#,:]
        else:
            #pdb.set_trace()
            # the best model result was on volt
            return spike #spike #spike # spike#[:,:,:30] #torch.mean(spike[:,:,:], axis=2) #250 got max 92% with .05e-2 meta-lr, but this was with an incorrect implementation I think
    
    
    def gen_loihi_params(self, folder):
        # Define Loihi parameter generator
        for i, b in enumerate(self.blocks):
            if hasattr(b, 'synapse'):
                if hasattr(b.synapse, 'weight'):
                    print(b.synapse.weight.cpu().data.numpy())
                    #weights = slayer.utils.quantize(b.synapse.weight, step=2).cpu().data.numpy() # this is not the same method used in training, probably why it's failing
                    #pdb.set_trace()
                    weights = b.synapse._pre_hook_fx(b.synapse.weight,descale=True).cpu().data.numpy()
                    weights = np.squeeze(weights)
                    print('Block: ' + str(b) + ' layer: ' + str(i) + ' with shape:' + str(np.shape(weights)))
                    print("max weight", np.max(weights))
                    print("layer weights", weights)
                    np.save(folder + '/' + f'{i}' + '_weights.npy', weights)
                if b.delay:
                    #delays = slayer.utils.quantize(b.delay.delay, step=2).cpu().data.numpy() # this is not the same method used in training, probably why it is failing
                    delays = b.synapse._pre_hook_fx(b.delay.delay,descale=True).cpu().data.numpy()
                    # delays = np.squeeze(delays, axis=-1)
                    np.save(folder + '/' + f'{i}' + '_delays.npy', delays)
                    print('Delay layer: ' + str(i) + ' with shape:' + str(np.shape(delays)))

    
    
    def transpose_torchneuromorphic_to_SLAYER(self, data_batch):
        data_batch = torch.transpose(data_batch, 1,2)
        data_batch = torch.transpose(data_batch,2,3)
        data_batch = torch.transpose(data_batch,3,4)
        # it looks like torchneuromorphic does y,x so switch them
        data_batch = torch.transpose(data_batch,2,3)
        return data_batch

    
    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

    def export_hdf5(self, filename):
        # network export to hdf5 format
        # compatable with netx for use with loihi 2 (1 could work too)
        # net = netx.hdf5.Network(net_config='filename', other params)
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))
            
    
        
        
class MetaLavaNet(MetaModuleNg):
    def __init__(self, params):
        super(MetaLavaNet, self).__init__()
        
        self.network_params = params['network']
        
        self.neuron_params = params['neuron_model']
        
        self.neuron_params_dropout = {**params['neuron_model'], 'dropout' : slayer.neuron.Dropout(p=0.0)} # was .05. from looking at the example input and hidden dense layers used this. Not sure about conv, would have to experiment I guess
        
        #pdb.set_trace()
        
        self.blocks = MetaSequential()
        
        self.input_given = False
        
        self.Mhid = []
        
        if self.network_params['Nhid']:
            conv_stack_output_shape = list(self.build_conv_stack())
            #pdb.set_trace()
            #conv_stack_output_shape[-1] += 1
            mlp_in = int(np.prod(conv_stack_output_shape))
            
            self.Mhid = [mlp_in] + self.network_params['Mhid']

        if self.network_params['Mhid']:
            mlp_stack_output_shape = self.build_mlp_stack()
            
            if not self.Mhid:
                self.Mhid = [mlp_stack_output_shape]

        if self.network_params['out_channels'] is not None:
            output_shape = self.build_output_layer()
            
        self.analog_readout=self.network_params['analog_readout'] # Emre defaults to True
        
        if self.analog_readout:
            self.blocks[-1].neuron.return_internal_state = True
            
        
        for block in self.blocks:
            block.neuron.quantize = self.neuron_params['quantize']
            
            
        # use data initialization
        
        
    def build_conv_stack(self):    
        # slayer.block.sigma_delta.Conv(sdnn_cnn_params,  in, out, kernel, padding=0, stride=2, weight_scale=2, weight_norm=True) just an example for reference
        i = 0
        
        feature_height = self.network_params['input_shape'][1]
        feature_width = self.network_params['input_shape'][2]
        
        for nhid in range(len(self.network_params['Nhid'])):
            
            padding = (self.network_params['kernel_size'][nhid] - 1) // 2
            
            if not self.input_given:
                self.blocks.append(slayer.block.cuba.MetaConv(self.neuron_params, self.network_params['input_shape'][0], self.network_params['Nhid'][nhid], kernel_size=self.network_params['kernel_size'][nhid], stride=self.network_params['stride'][nhid],padding=padding, weight_scale=1, weight_norm=False,delay=self.network_params['delay'],delay_shift=True,pre_hook_fx=None if self.neuron_params['quantize'] else lambda x: x))
                self.input_given = True
                # pool params: kernel_size, stride=None, padding=0, dilation=1, weight_scale=1, weight_norm=False, pre_hook_fx=None), remember pre_hook_fx is for quantization so I'll need to add this for loihi
                self.blocks.append(slayer.block.cuba.Pool(self.neuron_params,kernel_size=self.network_params['pool_size'][nhid],delay=self.network_params['delay'],delay_shift=False,pre_hook_fx=None if self.neuron_params['quantize'] else lambda x: x))
                
            else:
                self.blocks.append(slayer.block.cuba.MetaConv(self.neuron_params, self.network_params['Nhid'][nhid-1], self.network_params['Nhid'][nhid], kernel_size=self.network_params['kernel_size'][nhid], stride=self.network_params['stride'][nhid],padding=padding,weight_scale=1, weight_norm=False,delay=self.network_params['delay'],delay_shift=True,pre_hook_fx=None if self.neuron_params['quantize'] else lambda x: x))
                self.blocks.append(slayer.block.cuba.Pool(self.neuron_params,kernel_size=self.network_params['pool_size'][nhid],delay=self.network_params['delay'],delay_shift=False,pre_hook_fx=None if self.neuron_params['quantize'] else lambda x: x))
                
            
            feature_height, feature_width = self.get_output_shape(
                [feature_height, feature_width], 
                kernel_size = self.network_params['kernel_size'][nhid],
                stride = self.network_params['stride'][nhid],
                padding = (self.network_params['kernel_size'][nhid] - 1) // 2, #[1] if self.network_params['padding'][nhid] is None else [self.network_params['padding'][nhid]],
                dilation = 1)
            
            feature_height //= self.network_params['pool_size'][nhid]
            feature_width //= self.network_params['pool_size'][nhid]
                
        return (self.network_params['Nhid'][-1],feature_height, feature_width)
                                   
    def build_mlp_stack(self):
        # slayer.block.cuba.Dense(neuron_params, 34*34*2, 512, weight_norm=True, delay=True) example for reference
        for mhid in range(len(self.network_params['Mhid'])):
            if not self.input_given:
                self.blocks.append(slayer.block.cuba.MetaDense(self.neuron_params_dropout, int(np.prod(self.network_params['input_shape'])), self.network_params['Mhid'][mhid], weight_norm=False, delay=self.network_params['delay'],delay_shift=True,weight_scale=1,pre_hook_fx= None if self.neuron_params['quantize'] else lambda x: x))#,pre_hook_fx=lambda x: x))#pre_hook_fx=lambda x: x))
                self.input_given = True
            else:
                 self.blocks.append(slayer.block.cuba.MetaDense(self.neuron_params_dropout, self.network_params['Mhid'][mhid-1], self.network_params['Mhid'][mhid], weight_norm=False, delay=self.network_params['delay'],delay_shift=True,weight_scale=1, pre_hook_fx=None if self.neuron_params['quantize'] else lambda x: x))#,pre_hook_fx=lambda x: x))#pre_hook_fx=lambda x: x))
                    
        return (self.network_params['Mhid'][-1])

    # can try to set weight_norm to True, should act like batch norm which could help
    # try adding refractory dynamics, alif in lava has it (lif with adaptive threshold, could turn off for lif)
    # examine spike activity of the layers, average spike rate in the layers, ~10-50% ballpark
    # if spiking too much, lower initial weight values. Too little then raise. weight_scale param
    # make sure layers keep spiking and are not dying
    # make sure graidents of inner and outer loop are not vanishing or exploding
    
    def build_output_layer(self):
        self.blocks.append(slayer.block.cuba.MetaDense(self.neuron_params, self.Mhid[-1], self.network_params['out_channels'], weight_norm=False,delay=self.network_params['delay'],delay_shift=self.network_params['delay'], weight_scale=1, pre_hook_fx=None if self.neuron_params['quantize'] else lambda x: x))#,pre_hook_fx=lambda x: x))#,pre_hook_fx=None))
        return (self.network_params['out_channels']) #bias not compatible with loihi 1, but is with loihi2 with microcode. Bias can work on loihi 1 if the current decay is 4096, then it will work
    
    def get_output_shape(self, input_shape, kernel_size, stride, padding, dilation):
        #pdb.set_trace()
        # must be incorrect since dvssign doesn't work with cnn
        im_height = input_shape[-2]
        im_width  = input_shape[-1]
        
        height = int((im_height + 2 * padding - dilation *
                          (kernel_size - 1) - 1) // stride + 1)
        # if len(kernel_size)>1:
        #     width = int((im_width + 2 * padding - dilation *
        #                   (kernel_size - 1) - 1) // stride + 1)
        # else:
        width = int((im_width + 2 * padding - dilation *
                          (kernel_size - 1) - 1) // stride + 1)
            
        return [height, width]
        
#     self.blocks = torch.nn.ModuleList([
#             slayer.block.cuba.Dense(neuron_params, 34*34*2, 512, weight_norm=True, delay=True),
#             slayer.block.cuba.Dense(neuron_params_drop, 512, 512, weight_norm=True, delay=True),
#             slayer.block.cuba.Dense(neuron_params, 512, 10, weight_norm=True),
#         ])
    
    def forward(self, spike, params=None):
        #self.analog_readout=False
        #self.blocks[-1].neuron.return_internal_state = False
        #spike = self.transpose_torchneuromorphic_to_SLAYER(spike)
        
        spike_input = spike.permute(0,2,3,4,1)
        
        if not self.network_params['Nhid']:
            # first layer is mlp, flatten input
            spike = spike.flatten(1,3)
            # try something like burnin which skips a certain number of steps
            #spike = spike[:,:,70:]
            
        params_wb = OrderedDict()
            
        for k in range(1): # really simple way of doing a burnin type thing
            spike = spike_input
            i = 0
            j = 0
            for block in self.blocks:
                if self.network_params['Nhid']:
                    if i%2==0 and j!=(len(self.network_params['Nhid'])):
                        #params_wb['weight'] = params[f'blocks.{i}.synapse.weight']# params[f'blocks.{i}.synapse.weight_v']
                        #params_wb['bias'] = None #params[f'blocks.{i}.synapse.weight_g']
                        #pdb.set_trace()
                        spike, volt = block(spike, params=params[f'blocks.{i}.synapse.weight'])
                        j+=1
                    elif i==(len(self.blocks)-1):
                        # should be output layer, need to flatten input
                        spike = spike.flatten(1,3)
                        #params_wb['weight'] = params[f'blocks.{i}.synapse.weight']
                        #params_wb['bias'] = None
                        # params_wb['weight'] = params[f'blocks.{i}.synapse.weight_v']
                        # params_wb['bias'] = params[f'blocks.{i}.synapse.weight_g']
                        spike, volt = block(spike, params=params[f'blocks.{i}.synapse.weight'])
                    else:
                        # should be pooling layer
                        spike, volt = block(spike)


                else:
                    # if i==len(self.network_params['Mhid']):
                    #     params_wb['weight'] = params[f'blocks.{i}.synapse.weight']
                    #     params_wb['bias'] = None
                    # elif 'blocks.{i}.synapse.weight_v' in params.keys():
                    #     params_wb['weight'] = params[f'blocks.{i}.synapse.weight_v']
                    #     params_wb['bias'] = params[f'blocks.{i}.synapse.weight_g']
                    # else:
                    #     params_wb['weight'] = params[f'blocks.{i}.synapse.weight']
                    #     params_wb['bias'] = None

                    #pdb.set_trace()

                    if len(spike.shape)>3:
                        spike = spike.flatten(1,3)
                    if block.neuron.return_internal_state:
                        #pdb.set_trace()
                        spike, volt = block(spike, params[f'blocks.{i}.synapse.weight'])
                        #print("returning internal state")
                        #pdb.set_trace()
                    else:
                        #pdb.set_trace()
                        spike, volt = block(spike, params[f'blocks.{i}.synapse.weight'])
                        #print("before final layer")
                        #pdb.set_trace

                i+=1

        # in Emre's implementation he has this, which could be an important difference for how the performance is
        # his is also using torch's cross entropy loss instead of a slayer loss, which could also be a difference maker
        #pdb.set_trace()
        if self.analog_readout:
            #pdb.set_trace()
            #print("analog readout")
            #return torch.mean(curr_volt[1][:,:],axis=1)
            # spike
            return torch.sum(spike,axis=-1) #volt #torch.sum(volt[:,:],axis=2)#,-1]#-1 #[1][:,:]#,:]
        else:
            #pdb.set_trace()
            # THE BEST META LEARNING RESULT WAS DONE WITH volt AS THE OUTPUT with fastsigmoid!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # but this doesn't use spikes, which isn't compatible with soel.
            return volt #spike #volt # spike#[:,:,:30] #torch.mean(spike[:,:,:], axis=2) #250 got max 92% with .05e-2 meta-lr, but this was with an incorrect implementation I think
    
    
    def gen_loihi_params(self, folder):
        # Define Loihi parameter generator
        for i, b in enumerate(self.blocks):
            if hasattr(b, 'synapse'):
                if isinstance(b.synapse, Pool):
                    weights = b.synapse.weight.cpu().data.numpy()
                    weights = np.squeeze(weights)
                    print('Block: ' + str(b) + ' layer: ' + str(i) + ' with shape:' + str(np.shape(weights)))
                    print("max weight", np.max(weights))
                    print("layer weights", weights)
                    np.save(folder + '/' + f'pool_{i}' + '_weights.npy', weights)
                elif hasattr(b.synapse, 'weight'):
                    print(b.synapse.weight.cpu().data.numpy())
                    #weights = slayer.utils.quantize(b.synapse.weight, step=2).cpu().data.numpy() # this is not the same method used in training, probably why it's failing
                    #pdb.set_trace()
                    weights = b.synapse._pre_hook_fx(b.synapse.weight,descale=True).cpu().data.numpy()
                    weights = np.squeeze(weights)
                    print('Block: ' + str(b) + ' layer: ' + str(i) + ' with shape:' + str(np.shape(weights)))
                    print("max weight", np.max(weights))
                    print("layer weights", weights)
                    np.save(folder + '/' + f'{i}' + '_weights.npy', weights)
                if b.delay:
                    #delays = slayer.utils.quantize(b.delay.delay, step=2).cpu().data.numpy() # this is not the same method used in training, probably why it is failing
                    delays = b.synapse._pre_hook_fx(b.delay.delay,descale=True).cpu().data.numpy()
                    # delays = np.squeeze(delays, axis=-1)
                    np.save(folder + '/' + f'{i}' + '_delays.npy', delays)
                    print('Delay layer: ' + str(i) + ' with shape:' + str(np.shape(delays)))

    
    
    def transpose_torchneuromorphic_to_SLAYER(self, data_batch):
        data_batch = torch.transpose(data_batch, 1,2)
        data_batch = torch.transpose(data_batch,2,3)
        data_batch = torch.transpose(data_batch,3,4)
        # it looks like torchneuromorphic does y,x so switch them
        data_batch = torch.transpose(data_batch,2,3)
        return data_batch

    
    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

    def export_hdf5(self, filename):
        # network export to hdf5 format
        # compatable with netx for use with loihi 2 (1 could work too)
        # net = netx.hdf5.Network(net_config='filename', other params)
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))
            
            
class MetaLavaNetALIF(MetaLavaNet):
    def __init__(self, params):
        super(MetaLavaNetALIF, self).__init__(params)
        
        self.network_params = params['network']
        
        self.neuron_params = params['neuron_model']
        
        self.neuron_params_dropout = {**params['neuron_model'], 'dropout' : slayer.neuron.Dropout(p=0.0)} # was .05. from looking at the example input and hidden dense layers used this. Not sure about conv, would have to experiment I guess
        
        #pdb.set_trace()
        
        self.blocks = MetaSequential()
        
        self.input_given = False
        
        self.Mhid = []
        
        if self.network_params['Nhid']:
            conv_stack_output_shape = self.build_conv_stack()
            mlp_in = int(np.prod(conv_stack_output_shape))
            
            self.Mhid = [mlp_in] + self.network_params['Mhid']

        if self.network_params['Mhid']:
            mlp_stack_output_shape = self.build_mlp_stack()
            
            if not self.Mhid:
                self.Mhid = [mlp_stack_output_shape]

        if self.network_params['out_channels'] is not None:
            output_shape = self.build_output_layer()
            
        self.analog_readout=self.network_params['analog_readout'] # Emre defaults to True
        
        if self.analog_readout:
            self.blocks[-1].neuron.return_internal_state = True
            
        
        for block in self.blocks:
            block.neuron.quantize = self.neuron_params['quantize']
            
            
    def build_conv_stack(self):    
        # slayer.block.sigma_delta.Conv(sdnn_cnn_params,  in, out, kernel, padding=0, stride=2, weight_scale=2, weight_norm=True) just an example for reference
        i = 0
        
        feature_height = self.network_params['input_shape'][1]
        feature_width = self.network_params['input_shape'][2]
        
        for nhid in range(len(self.network_params['Nhid'])):
            
            padding = (self.network_params['kernel_size'][nhid] - 1) // 2
            
            if not self.input_given:
                self.blocks.append(slayer.block.alif.MetaConv(self.neuron_params, self.network_params['input_shape'][0], self.network_params['Nhid'][nhid], kernel_size=self.network_params['kernel_size'][nhid], stride=self.network_params['stride'][nhid],padding=padding, weight_scale=1, weight_norm=False,delay=self.network_params['delay'],delay_shift=False,pre_hook_fx=None if self.neuron_params['quantize'] else lambda x: x))
                self.input_given = True
                # pool params: kernel_size, stride=None, padding=0, dilation=1, weight_scale=1, weight_norm=False, pre_hook_fx=None), remember pre_hook_fx is for quantization so I'll need to add this for loihi
                self.blocks.append(slayer.block.alif.Pool(self.neuron_params,kernel_size=self.network_params['pool_size'][nhid],delay=self.network_params['delay'],delay_shift=True,pre_hook_fx=None if self.neuron_params['quantize'] else lambda x: x))
                
            else:
                self.blocks.append(slayer.block.alif.MetaConv(self.neuron_params, self.network_params['Nhid'][nhid-1], self.network_params['Nhid'][nhid], kernel_size=self.network_params['kernel_size'][nhid], stride=self.network_params['stride'][nhid],padding=padding,weight_scale=1, weight_norm=False,delay=self.network_params['delay'],delay_shift=False,pre_hook_fx=None if self.neuron_params['quantize'] else lambda x: x))
                self.blocks.append(slayer.block.alif.Pool(self.neuron_params,kernel_size=self.network_params['pool_size'][nhid],delay=self.network_params['delay'],delay_shift=True,pre_hook_fx=None if self.neuron_params['quantize'] else lambda x: x))
                
            feature_height, feature_width = self.get_output_shape(
                [feature_height, feature_width], 
                kernel_size = self.network_params['kernel_size'][nhid],
                stride = self.network_params['stride'][nhid],
                padding = (self.network_params['kernel_size'][nhid] - 1) // 2, #[1] if self.network_params['padding'][nhid] is None else [self.network_params['padding'][nhid]],
                dilation = 1)
            feature_height //= self.network_params['pool_size'][nhid]
            feature_width //= self.network_params['pool_size'][nhid]
                
        return (self.network_params['Nhid'][-1],feature_height, feature_width)
                                   
    def build_mlp_stack(self):
        # slayer.block.cuba.Dense(neuron_params, 34*34*2, 512, weight_norm=True, delay=True) example for reference
        for mhid in range(len(self.network_params['Mhid'])):
            if not self.input_given:
                self.blocks.append(slayer.block.alif.MetaDense(self.neuron_params_dropout, int(np.prod(self.network_params['input_shape'])), self.network_params['Mhid'][mhid], weight_norm=False, delay=self.network_params['delay'],delay_shift=self.network_params['delay'],weight_scale=1,pre_hook_fx= None if self.neuron_params['quantize'] else lambda x: x))#,pre_hook_fx=lambda x: x))#pre_hook_fx=lambda x: x))
                self.input_given = True
            else:
                 self.blocks.append(slayer.block.alif.MetaDense(self.neuron_params_dropout, self.network_params['Mhid'][mhid-1], self.network_params['Mhid'][mhid], weight_norm=False, delay=self.network_params['delay'],delay_shift=self.network_params['delay'],weight_scale=1, pre_hook_fx=None if self.neuron_params['quantize'] else lambda x: x))#,pre_hook_fx=lambda x: x))#pre_hook_fx=lambda x: x))
                    
        return (self.network_params['Mhid'][-1])

    # can try to set weight_norm to True, should act like batch norm which could help
    # try adding refractory dynamics, alif in lava has it (lif with adaptive threshold, could turn off for lif)
    # examine spike activity of the layers, average spike rate in the layers, ~10-50% ballpark
    # if spiking too much, lower initial weight values. Too little then raise. weight_scale param
    # make sure layers keep spiking and are not dying
    # make sure graidents of inner and outer loop are not vanishing or exploding
    
    def build_output_layer(self):
        self.blocks.append(slayer.block.alif.MetaDense(self.neuron_params, self.Mhid[-1], self.network_params['out_channels'], weight_norm=False,delay=self.network_params['delay'],delay_shift=self.network_params['delay'], weight_scale=1, pre_hook_fx=None if self.neuron_params['quantize'] else lambda x: x))#,pre_hook_fx=lambda x: x))#,pre_hook_fx=None))
        return (self.network_params['out_channels'])
        

def build_model_lava(out_features, params_file, device, detach_at, sg_function_baseline=False): # = 'parameters/lava_params_dnmnist.yml'
    from decolle.utils import parse_args, prepare_experiment, cross_entropy_one_hot
    import torch
    import torch.nn.functional as F

    params_file = params_file 
    with open(params_file, 'r') as f:
        import yaml
        params = yaml.safe_load(f)
    verbose = True
    
    # params['neuron_model']['current_decay'] = round(random.uniform(0.01,0.99),2)
    # params['neuron_model']['voltage_decay'] = round(random.uniform(0.01,0.99),2)
    
    # with open(params_file, 'w') as outfile:
    #     yaml.dump(params, outfile)
    
    if params['alif']:
        print("USING ALIF NEURON")
        net = MetaLavaNetALIF(params).to(device)
    else:
        net = MetaLavaNet(params).to(device)
    
    print(net) # make sure it made the proper network
    
    #pdb.set_trace()
    
    # decolle adds gain to the input lif layer, not sure what the equivalent is for lava-dl or if it'll help
    # there's also parameter initialization that decolle does. Will the function work for slayer??? no I would need to add it or call what slayer uses
    #net.init_parameters(torch.zeros([1,params['network']['chunk_size_train']]+params['network']['input_shape']).to(device))

    return net
    
