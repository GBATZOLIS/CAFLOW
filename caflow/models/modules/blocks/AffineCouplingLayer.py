#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 13:29:34 2021

@author: gbatz97
"""

"""
The main code for the CouplingLayer has been copied from a deep learning tutorial on normalising flows in the University of Amsterdam
Tutorial: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html
"""

import torch
import torch.nn as nn

class AffineCouplingLayer(nn.Module):
    def __init__(self, c_in, dim, mask_info, network):
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

        super(AffineCouplingLayer, self).__init__()
        self.c_in = c_in
        self.dim = dim
        self.network = network

        # For stabilization purposes, we apply a tanh activation function on the scaling output.
        # This prevents sudden large output values for the scaling that can destabilize training.
        # To still allow scaling factors smaller or larger than -1 and 1 respectively, 
        # we have a learnable parameter per dimension, called scaling_factor
        self.scaling_factor = nn.Parameter(torch.zeros(c_in)) #learnable scaling factor for each dimension

        # Register mask as buffer as it is a tensor which is not a parameter,
        # but should be part of the modules state.
        mask = self.create_mask(**mask_info)
        self.register_buffer('mask', mask)

    def create_mask(self, **kwargs):
        """
        output: mask (Binary mask (0 or 1) where 0 denotes that the element should be transformed,
                      while 1 means the latent will be used as input to the NN._
        """
        if kwargs.get('mask_type') == 'channel':
            c_in = self.c_in
            mask = torch.cat([torch.ones(c_in//2, dtype=torch.float32),
                            torch.zeros(c_in-c_in//2, dtype=torch.float32)])
            infered_shape = tuple([1,c_in]+[1 for i in range(self.dim)])
            mask = mask.view(infered_shape)
            
            if kwargs.get('invert')==True:
                mask = 1 - mask
                
            return mask
        
        elif kwargs.get('mask_type') == 'checkerboard':
            shape = kwargs.get('shape')
            dims = []
            for dim_size in shape:
                dims.append[torch.arange(dim_size, dtype=torch.int32)]
            mesh_dims = torch.meshgrid(*dims)
            mesh_dims_sum = torch.sum(mesh_dims, dim=0)
            mask = torch.fmod(mesh_dims_sum, 2)
            infered_shape = tuple([1,1]+[dim for dim in shape])
            mask = mask.to(torch.float32).view(infered_shape)
            
            if kwargs.get('invert')==True:
                mask = 1 - mask
                
            return mask
        
        else:
            raise NotImplementedError('mask type has not been implemented')

    def forward(self, z, logdet, reverse=False, cond_rv=[]):
        """
        Inputs:
            z - Latent input to the flow
            ldj - The current ldj of the previous flows.
                  The ldj of this layer will be added to this tensor.
            reverse - If True, we apply the inverse of the layer.
            cond_rv (optional) - Allows external input to condition the flow on
        """

        # Apply network to masked input
        z_in = z * self.mask
        
        if not cond_rv:
            nn_out = self.network(z_in)
        else:
            nn_out = self.network(z = z_in, cond_rv = cond_rv)
        
        #get the scaling and translation tensors
        s, t = nn_out.chunk(2, dim=1)

        # Stabilize scaling output
        infer_shape = tuple([1,-1]+[1 for i in range(self.dim)])
        s_fac = self.scaling_factor.exp().view(infer_shape)
        s = torch.tanh(s / s_fac) * s_fac

        # Mask outputs (only transform the second part)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        # Affine transformation
        if not reverse:
            # Whether we first shift and then scale, or the other way round,
            # is a design choice, and usually does not have a big impact
            z = (z + t) * torch.exp(s)
            #print('logdet.shape: ', logdet)
            #print('s.shape: ', s.shape)
            logdet += s.sum(dim=[i+1 for i in range(self.dim+1)]) #use self.dim+1 because we need to add over the channel dimension as well
        else:
            z = (z * torch.exp(-s)) - t
            #print(s.sum(dim=[i+1 for i in range(self.dim+1)]))
            #print(logdet)
            logdet -= s.sum(dim=[i+1 for i in range(self.dim+1)])

        return z, logdet