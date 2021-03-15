#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 19:28:47 2021

@author: gbatz97
"""



import torch.nn as nn
from torch.nn import Conv1d, Conv2d, Conv3d
import torch

class CondSimpleConvNet(nn.Module):

    def __init__(self, c_in, dim, c_hidden=32, c_out=-1, num_layers=1, layer_type='coupling', num_cond_rvs=2 , last_scale=False, interpolation=True):
        """
        Module that summarizes the previous blocks to a full convolutional neural network.
        Inputs:
            c_in - Number of input channels
            c_hidden - Number of hidden dimensions to use within the network
            c_out - Number of output channels. If -1, 2 times the input channels are used (affine coupling)
            num_layers - Number of gated ResNet blocks to apply
        """
        super(CondSimpleConvNet, self).__init__()
        
        self.layer_type = layer_type
        self.interpolation = interpolation

        conv = [Conv1d, Conv2d, Conv3d][dim-1] #select the appropriate conv layer based on the dimension of the input tensor
        c_out = c_out if c_out > 0 else 2 * c_in
    
        #reshape conditional rvs to the same shape as z using convolution if interpolation is set to True
        if self.interpolation:
            self.interpolate_layers = nn.ModuleList()
            for i in range(num_cond_rvs):
                if not last_scale:
                    self.interpolate_layers.append(conv(in_channels=c_in//2**dim, out_channels=c_in, 
                                                        kernel_size = 4, stride = 2, padding = 1, dilation = 1))
                else:
                    self.interpolate_layers.append(conv(in_channels=c_in//2**(dim-1), out_channels=c_in, 
                                                        kernel_size = 4, stride = 2, padding = 1, dilation = 1))
                    
            
        
        #main network
        layers = nn.ModuleList()
        
        if self.layer_type =='injector':
            input_channels = num_cond_rvs * c_in
        elif self.layer_type == 'coupling':
            input_channels = (1 + num_cond_rvs) * c_in
        else:
            raise NotImplementedError('This type of layer is not supported yet. Options: [coupling, injector]')
            
        layers += [conv(input_channels, c_hidden, kernel_size=3, padding=1), nn.ReLU(inplace=False)]
        for layer_index in range(num_layers):
            layers += [conv(c_hidden, c_hidden, kernel_size=1),
                       nn.ReLU(inplace=False)]
            
        layers += [conv(c_hidden, c_out, kernel_size=3, padding=1)]
        
        self.nn = nn.Sequential(*layers)
        self.nn[-1].weight.data.zero_()
        self.nn[-1].bias.data.zero_()
        
    def forward(self, z=None, cond_rv=[]):
        if self.interpolation:
            interpolated_cond_rvs = []
            for i, rv in enumerate(cond_rv):
                interpolated_cond_rvs.append(self.interpolate_layers[i](rv))
        else:
            interpolated_cond_rvs = cond_rv
        
        if self.layer_type == 'injector':
            concat_pass = torch.cat(interpolated_cond_rvs, dim=1)
            output = self.nn(concat_pass) 
            
        elif self.layer_type == 'coupling':
            batch_size = interpolated_cond_rvs[0].shape[0]
            gain_factor = z.shape[0] // batch_size
            
            try:
                #if GPU memory is not an issue we can this code snippet.
                interpolated_cond_rvs = torch.cat(interpolated_cond_rvs, dim=1)
                repeat_tuple = tuple([gain_factor]+[1 for i in range(len(interpolated_cond_rvs.shape)-1)])
                interpolated_cond_rvs = interpolated_cond_rvs.repeat(repeat_tuple)
                concat_pass = torch.cat([z, interpolated_cond_rvs], dim=1)
                output = self.nn(concat_pass)
            
            except RuntimeError as err:
                print(err)
                
                #if memory is an issue we will use this code snippet
                
                outputs = []
                for i in range(gain_factor):
                    concat_pass = torch.cat([z[i*batch_size:(i+1)*batch_size]]+interpolated_cond_rvs, dim=1)
                    output = self.nn(concat_pass)
                    outputs.append(output)
                output = torch.cat(outputs, dim = 0)

            return output
            
        else:
            raise NotImplementedError('This type of layer is not supported yet. Options: [coupling, injector]')
            
         
        return output
