#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 18:09:39 2021

@author: gbatz97
"""


import torch.nn as nn
from torch.nn import Conv1d, Conv2d, Conv3d

class SimpleConvNet(nn.Module):

    def __init__(self, c_in, dim, c_hidden=32, c_out=-1, num_layers=1):
        """
        Module that summarizes the previous blocks to a full convolutional neural network.
        Inputs:
            c_in - Number of input channels
            c_hidden - Number of hidden dimensions to use within the network
            c_out - Number of output channels. If -1, 2 times the input channels are used (affine coupling)
            num_layers - Number of gated ResNet blocks to apply
        """
        super(SimpleConvNet, self).__init__()
        
        conv = [Conv1d, Conv2d, Conv3d][dim-1] #select the appropriate conv layer based on the dimension of the input tensor
        c_out = c_out if c_out > 0 else 2 * c_in
        
        layers = nn.ModuleList()
        layers += [conv(c_in, c_hidden, kernel_size=3, padding=1), nn.ReLU(inplace=False)]
        for layer_index in range(num_layers):
            layers += [conv(c_hidden, c_hidden, kernel_size=1),
                       nn.ReLU(inplace=False)]
            
        layers += [conv(c_hidden, c_out, kernel_size=3, padding=1)]
        
        self.nn = nn.Sequential(*layers)
        
        #start the coupling layer as identity -> t=0, exp(s)=1
        self.nn[-1].weight.data.zero_()
        self.nn[-1].bias.data.zero_()

    def forward(self, x):
        return self.nn(x)
