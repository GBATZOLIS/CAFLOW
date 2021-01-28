#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 18:18:51 2021

@author: gbatz97
"""

""""
This model has been copied from the deep learning tutorial on normalising flows in the University of Amsterdam
Tutorial: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html

Their explanation for using this network is the following:
    
As a last aspect of coupling layers, we need to decide for the deep neural network we want to apply in the coupling layers. 
The input to the layers is an image, and hence we stick with a CNN. 
Because the input to a transformation depends on all transformations before, 
it is crucial to ensure a good gradient flow through the CNN back to the input, 
which can be optimally achieved by a ResNet-like architecture. 
Specifically, we use a Gated ResNet that adds a sigma-gate to the skip connection, 
similarly to the input gate in LSTMs. The details are not necessarily important here, 
and the network is strongly inspired from Flow++ [3] in case you are interested in building even stronger models.
"""

import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, Conv2d, Conv3d
import torch


class ConcatELU(nn.Module):
    """
    Activation function that applies ELU in both direction (inverted and plain).
    Allows non-linearity while providing strong gradients for any input (important for final convolution)
    """

    def forward(self, x):
        return torch.cat([F.elu(x), F.elu(-x)], dim=1)


class LayerNormChannels(nn.Module):

    def __init__(self, c_in, dim):
        """
        This module applies layer norm across channels in an image. Has been shown to work well with ResNet connections.
        Inputs:
            c_in - Number of channels of the input
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(c_in)
        
        #we need the dimension of the tensor to compute the correct permutation so that we can use nn.layer_norm
        self.dim = dim 
    
    def permute(self, x, inverse=False):
        #this function permutes the dimensions of tensor x
        #In the forward permute, the channel axis is moved to the last axis (this is needed for the subsequent LayerNorm layer)
        #In the inverse permute, the axes return back to the initial positions
        #Normal arrangement: [B, C, H, W, Z]
        #Forward permute:    [B, H, W, Z, C]
        #Inverse permute:    [B, C, H, W, Z]
        
        if inverse == False:
            permute_axes = [0]+[(i+1)+1 for i in range(self.dim)]+[1]
        else:
            permute_axes = [0]+[self.dim+1]+[i+1 for i in range(self.dim)]
        
        x = x.permute(*permute_axes)
        return x
        
    def forward(self, x):
        x = self.permute(x, inverse=False)
        x = self.layer_norm(x)
        x = self.permute(x, inverse=True)
        return x


class GatedConv(nn.Module):

    def __init__(self, c_in, c_hidden, dim):
        """
        This module applies a two-layer convolutional ResNet block with input gate
        Inputs:
            c_in - Number of channels of the input
            c_hidden - Number of hidden dimensions we want to model (usually similar to c_in)
        """
        super(GatedConv, self).__init__()
        
        conv = [Conv1d, Conv2d, Conv3d][dim-1]
        
        self.net = nn.Sequential(
            conv(c_in, c_hidden, kernel_size=3, padding=1),
            ConcatELU(),
            conv(2*c_hidden, 2*c_in, kernel_size=1)
        )

    def forward(self, x):
        out = self.net(x)
        val, gate = out.chunk(2, dim=1)
        return x + val * torch.sigmoid(gate)


class GatedConvNet(nn.Module):

    def __init__(self, c_in, dim, c_hidden=32, c_out=-1, num_layers=3):
        """
        Module that summarizes the previous blocks to a full convolutional neural network.
        Inputs:
            c_in - Number of input channels
            c_hidden - Number of hidden dimensions to use within the network
            c_out - Number of output channels. If -1, 2 times the input channels are used (affine coupling)
            num_layers - Number of gated ResNet blocks to apply
        """
        super(GatedConvNet, self).__init__()
        
        conv = [Conv1d, Conv2d, Conv3d][dim-1] #select the appropriate conv layer based on the dimension of the input tensor
        c_out = c_out if c_out > 0 else 2 * c_in
        
        layers = []
        layers += [conv(c_in, c_hidden, kernel_size=3, padding=1)]
        for layer_index in range(num_layers):
            layers += [GatedConv(c_hidden, c_hidden, dim),
                       LayerNormChannels(c_hidden, dim)]
        layers += [ConcatELU(),
                   conv(2*c_hidden, c_out, kernel_size=3, padding=1)]
        
        self.nn = nn.Sequential(*layers)
        
        #start the coupling layer as identity -> t=0, exp(s)=1
        self.nn[-1].weight.data.zero_()
        self.nn[-1].bias.data.zero_()

    def forward(self, x):
        return self.nn(x)