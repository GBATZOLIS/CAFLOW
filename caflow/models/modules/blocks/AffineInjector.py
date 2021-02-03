#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 07:44:42 2021

@author: gbatz97
"""


import torch
import torch.nn as nn

class AffineInjector(nn.Module):
    def __init__(self, c_in, dim, network):
        """
        Coupling layer inside a normalizing flow.
        Inputs:
            c_in - Number of input channels
            dim = Number of dimensions of the input tensor
            network - A PyTorch nn.Module constituting the deep neural network for mu and sigma.
                      Output shape should be twice the channel size as the input, because of the masking system
            mask_info - Dict. It provides all the information for building the mask.
                        keywords: 'mask_type'. Value Options = ['channel','checkerboard']
                                  'shape'. Value: shape of the input tensor. This is needed only if mask_type = 'checkerboard'
                                  'invert'. Value: Boolean. Whether to invert the mask. This can be used instead of permutation layers.
        """

        super(AffineInjector, self).__init__()
        self.c_in = c_in
        self.dim = dim
        self.network = network

        # For stabilization purposes, we apply a tanh activation function on the scaling output.
        # This prevents sudden large output values for the scaling that can destabilize training.
        # To still allow scaling factors smaller or larger than -1 and 1 respectively, 
        # we have a learnable parameter per dimension, called scaling_factor
        self.scaling_factor = nn.Parameter(torch.zeros(c_in)) #learnable scaling factor for each dimension

    def forward(self, z, logdet, reverse=False, cond_rv=[]):
        """
        Inputs:
            z - Latent input to the flow
            ldj - The current ldj of the previous flows.
                  The ldj of this layer will be added to this tensor.
            reverse - If True, we apply the inverse of the layer.
            cond_rv (optional) - Allows external input to condition the flow on
        """

        
        nn_out = self.network(z=None, cond_rv = cond_rv)
        
        #get the scaling and translation tensors
        s, t = nn_out.chunk(2, dim=1)

        # Stabilize scaling output
        infer_shape = tuple([1,-1]+[1 for i in range(self.dim)])
        s_fac = self.scaling_factor.exp().view(infer_shape)
        s = torch.tanh(s / s_fac) * s_fac
        
        # The following code snippet is used only in the shortcut approach.
        # Check if nn_out has higher dimension 0 than s and t. 
        # If so, check if nn_out.shape[0] is a multiple of s.shape[0]
        # The factor by which nn_out.shape[0] is greater than s.shape[0] 
        # is the number of times s and t should be copied and concatenated in the first dimension
        # This is used in the shortcut approaches where we do not recompute activations, by computing them only once.
        if z.shape[0] > s.shape[0]:
            #print(z.shape[0], s.shape[0])
            assert z.shape[0] % s.shape[0] == 0, 'z.shape[0] is not a multiple of s.shape[0].'
            
            num_copies = z.shape[0]//s.shape[0]
            repeat_tuple = tuple([num_copies]+[1 for i in range(len(s.shape)-1)])
            s = s.repeat(repeat_tuple)
            t = t.repeat(repeat_tuple)
            

        # Affine transformation
        if not reverse:
            # Whether we first shift and then scale, or the other way round,
            # is a design choice, and usually does not have a big impact
            z = (z + t) * torch.exp(s)
            logdet += s.sum(dim=[i+1 for i in range(self.dim+1)]) #use self.dim+1 because we need to add over the channel dimension as well
        else:
            z = (z * torch.exp(-s)) - t
            logdet -= s.sum(dim=[i+1 for i in range(self.dim+1)])

        return z, logdet