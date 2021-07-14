#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 18:09:39 2021

@author: gbatz97
"""


import torch.nn as nn
from torch.nn import Conv1d, Conv2d, Conv3d, BatchNorm1d, BatchNorm2d, BatchNorm3d, LayerNorm
from torch.nn.utils import spectral_norm
class SimpleConvNet(nn.Module):
    def __init__(self, c_in, c_out, c_hidden_factor, dim, resolution):
        """
        Module that summarizes the previous blocks to a full convolutional neural network.
        Inputs:
            c_in - Number of input channels
            c_hidden - Number of hidden dimensions to use within the network
            c_out - Number of output channels. If -1, 2 times the input channels are used (affine coupling)
            num_layers - Number of gated ResNet blocks to apply
        """
        super(SimpleConvNet, self).__init__()
        
        conv = [Conv1d, Conv2d, Conv3d][dim - 1]

        #c_hidden = int(c_hidden_factor * c_in)
        c_hidden = c_hidden_factor
        
        layers = nn.ModuleList()
        
        layers += [spectral_norm(conv(c_in, c_hidden, kernel_size=3, padding=1)), nn.ReLU(inplace=False)]
        for _ in range(1):
            layers += [spectral_norm(conv(c_hidden, c_hidden, kernel_size=1)), nn.ReLU(inplace=False)]
        layers += [conv(c_hidden, c_out, kernel_size=3, padding=1)]

        self.nn = nn.Sequential(*layers)
        
        #start the coupling layer as identity -> t=0, exp(s)=1
        #self.nn[-1].weight.data.zero_()
        #self.nn[-1].bias.data.zero_()

    def forward(self, x):
        return self.nn(x)
