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

from typing import Callable, Union
import torch
import torch.nn as nn
from FrEIA.modules import InvertibleModule 

class _BaseCouplingBlock(InvertibleModule):
    '''Base class to implement various coupling schemes.  It takes care of
    checking the dimensions, conditions, clamping mechanism, etc.
    Each child class only has to implement the _coupling1 and _coupling2 methods
    for the left and right coupling operations.
    (In some cases below, forward() is also overridden)
    '''

    def __init__(self, dims_in, dims_c=[],
                 clamp: float = 2.,
                 clamp_activation: Union[str, Callable] = "TANH"):
        '''
        Additional args in docstring of base class.
        Args:
          clamp: Soft clamping for the multiplicative component. The
            amplification or attenuation of each input dimension can be at most
            exp(±clamp).
          clamp_activation: Function to perform the clamping. String values
            "ATAN", "TANH", and "SIGMOID" are recognized, or a function of
            object can be passed. TANH behaves like the original realNVP paper.
            A custom function should take tensors and map -inf to -1 and +inf to +1.
        '''

        super().__init__(dims_in, dims_c)

        self.channels = dims_in[0][0]

        # ndims means the rank of tensor strictly speaking.
        # i.e. 1D, 2D, 3D tensor, etc.
        self.ndims = len(dims_in[0])

        self.split_len1 = self.channels // 2
        self.split_len2 = self.channels - self.channels // 2

        self.clamp = clamp

        assert all([tuple(dims_c[i][1:]) == tuple(dims_in[0][1:]) for i in range(len(dims_c))]), \
            "Dimensions of input and one or more conditions don't agree."
        self.conditional = (len(dims_c) > 0)
        self.condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        if isinstance(clamp_activation, str):
            if clamp_activation == "ATAN":
                self.f_clamp = (lambda u: 0.636 * torch.atan(u))
            elif clamp_activation == "TANH":
                self.f_clamp = torch.tanh
            elif clamp_activation == "SIGMOID":
                self.f_clamp = (lambda u: 2. * (torch.sigmoid(u) - 0.5))
            else:
                raise ValueError(f'Unknown clamp activation "{clamp_activation}"')
        else:
            self.f_clamp = clamp_activation

    def forward(self, x, c=[], rev=False, jac=True):
        '''See base class docstring'''

        # notation:
        # x1, x2: two halves of the input
        # y1, y2: two halves of the output
        # *_c: variable with condition concatenated
        # j1, j2: Jacobians of the two coupling operations

        x1, x2 = torch.split(x[0], [self.split_len1, self.split_len2], dim=1)

        if not rev:
            x2_c = torch.cat([x2, *c], 1) if self.conditional else x2
            y1, j1 = self._coupling1(x1, x2_c)

            y1_c = torch.cat([y1, *c], 1) if self.conditional else y1
            y2, j2 = self._coupling2(x2, y1_c)
        else:
            # names of x and y are swapped for the reverse computation
            x1_c = torch.cat([x1, *c], 1) if self.conditional else x1
            y2, j2 = self._coupling2(x2, x1_c, rev=True)

            y2_c = torch.cat([y2, *c], 1) if self.conditional else y2
            y1, j1 = self._coupling1(x1, y2_c, rev=True)

        return (torch.cat((y1, y2), 1),), j1 + j2

    def _coupling1(self, x1, u2, rev=False):
        '''The first/left coupling operation in a two-sided coupling block.
        Args:
          x1 (Tensor): the 'active' half being transformed.
          u2 (Tensor): the 'passive' half, including the conditions, from
            which the transformation is computed.
        Returns:
          y1 (Tensor): same shape as x1, the transformed 'active' half.
          j1 (float or Tensor): the Jacobian, only has batch dimension.
            If the Jacobian is zero of fixed, may also return float.
        '''
        raise NotImplementedError()

    def _coupling2(self, x2, u1, rev=False):
        '''The second/right coupling operation in a two-sided coupling block.
        Args:
          x2 (Tensor): the 'active' half being transformed.
          u1 (Tensor): the 'passive' half, including the conditions, from
            which the transformation is computed.
        Returns:
          y2 (Tensor): same shape as x1, the transformed 'active' half.
          j2 (float or Tensor): the Jacobian, only has batch dimension.
            If the Jacobian is zero of fixed, may also return float.
        '''
        raise NotImplementedError()

    def output_dims(self, input_dims):
        '''See base class for docstring'''
        if len(input_dims) != 1:
            raise ValueError("Can only use 1 input")
        return input_dims

class AffineCouplingOneSided(_BaseCouplingBlock):
    '''Half of a coupling block following the GLOWCouplingBlock design.  This
    means only one affine transformation on half the inputs.  In the case where
    random permutations or orthogonal transforms are used after every block,
    this is not a restriction and simplifies the design.  '''

    def __init__(self, dims_in, dims_c=[],
                    subnet_constructor: Callable = None,
                    clamp: float = 2.,
                    clamp_activation: Union[str, Callable] = "ATAN", nn_settings=None):
        '''
            Additional args in docstring of base class.
            Args:
            subnet_constructor: function or class, with signature
                constructor(dims_in, dims_out).  The result should be a torch
                nn.Module, that takes dims_in input channels, and dims_out output
                channels. See tutorial for examples. One subnetwork will be
                initialized in the block.
            clamp: Soft clamping for the multiplicative component. The
                amplification or attenuation of each input dimension can be at most
                exp(±clamp).
            clamp_activation: Function to perform the clamping. String values
                "ATAN", "TANH", and "SIGMOID" are recognized, or a function of
                object can be passed. TANH behaves like the original realNVP paper.
                A custom function should take tensors and map -inf to -1 and +inf to +1.
        '''

        super().__init__(dims_in, dims_c, clamp, clamp_activation)
        nn_type = nn_settings['nn_type']
        if nn_type == 'SimpleConvNet':
            c_hidden_factor = nn_settings['c_hidden_factor']
            dim = nn_settings['dim']
            self.subnet = subnet_constructor(c_in=self.split_len1 + self.condition_length, \
                                        c_out = 2 * self.split_len2, c_hidden_factor=c_hidden_factor, dim=dim, resolution = dims_in[0][1:])
        elif nn_type == 'nnflowpp':
            coupling = 'Affine'
            in_channels = self.split_len1 + self.condition_length
            out_channels = 2 * self.split_len2
            num_channels_factor = nn_settings['num_channels_factor']
            num_blocks = nn_settings['num_blocks']
            drop_prob = nn_settings['drop_prob']
            use_attn = nn_settings['use_attn']
            #aux_channels = nn_settings['aux_channels']
            self.subnet = subnet_constructor(coupling, in_channels, out_channels, num_channels_factor, num_blocks, drop_prob, use_attn)

    def forward(self, x, c=[], rev=False, jac=True):
        x1, x2 = torch.split(x[0], [self.split_len1, self.split_len2], dim=1)
        if self.conditional:
            if x1.size(0) > c[0].size(0):
                gain_factor = x1.size(0) // c[0].size(0)
                repeat_tuple = tuple([gain_factor]+[1 for i in range(len(c[0].shape)-1)])
                c = torch.cat(c, 1).repeat(repeat_tuple)
                x1_c = torch.cat([x1, c], 1)
            elif x1.size(0) == c[0].size(0):
                x1_c = torch.cat([x1, *c], 1)
            else:
                raise NotImplementedError
        else:
            x1_c = x1

        # notation:
        # x1, x2: two halves of the input
        # y1, y2: two halves of the output
        # a: all affine coefficients
        # s, t: multiplicative and additive coefficients
        # j: log det Jacobian

        a = self.subnet(x1_c)
        s, t = a[:, :self.split_len2], a[:, self.split_len2:]
        s = self.clamp * self.f_clamp(s)
        j = torch.sum(s, dim=tuple(range(1, self.ndims + 1)))

        if rev:
            y2 = (x2 - t) * torch.exp(-s)
            j *= -1
        else:
            y2 = x2 * torch.exp(s) + t

        return (torch.cat((x1, y2), 1),), j

class AdditiveCouplingOneSided(_BaseCouplingBlock):
    '''Half of a coupling block following the GLOWCouplingBlock design.  This
    means only one affine transformation on half the inputs.  In the case where
    random permutations or orthogonal transforms are used after every block,
    this is not a restriction and simplifies the design.  '''

    def __init__(self, dims_in, dims_c=[],
                    subnet_constructor: Callable = None,
                    clamp: float = 2.,
                    clamp_activation: Union[str, Callable] = "ATAN", nn_settings=None):
        '''
            Additional args in docstring of base class.
            Args:
            subnet_constructor: function or class, with signature
                constructor(dims_in, dims_out).  The result should be a torch
                nn.Module, that takes dims_in input channels, and dims_out output
                channels. See tutorial for examples. One subnetwork will be
                initialized in the block.
            clamp: Soft clamping for the multiplicative component. The
                amplification or attenuation of each input dimension can be at most
                exp(±clamp).
            clamp_activation: Function to perform the clamping. String values
                "ATAN", "TANH", and "SIGMOID" are recognized, or a function of
                object can be passed. TANH behaves like the original realNVP paper.
                A custom function should take tensors and map -inf to -1 and +inf to +1.
        '''

        super().__init__(dims_in, dims_c, clamp, clamp_activation)
        nn_type = nn_settings['nn_type']
        if nn_type == 'SimpleConvNet':
            c_hidden_factor = nn_settings['c_hidden_factor']
            dim = nn_settings['dim']
            self.subnet = subnet_constructor(c_in=self.split_len1 + self.condition_length, \
                                        c_out = self.split_len2, c_hidden_factor=c_hidden_factor, dim=dim, resolution = dims_in[0][1:])
        elif nn_type == 'nnflowpp':
            coupling = 'Affine'
            in_channels = self.split_len1 + self.condition_length
            out_channels = self.split_len2
            num_channels_factor = nn_settings['num_channels_factor']
            num_blocks = nn_settings['num_blocks']
            drop_prob = nn_settings['drop_prob']
            use_attn = nn_settings['use_attn']
            #aux_channels = nn_settings['aux_channels']
            self.subnet = subnet_constructor(coupling, in_channels, out_channels, num_channels_factor, num_blocks, drop_prob, use_attn)

    def forward(self, x, c=[], rev=False, jac=True):
        x1, x2 = torch.split(x[0], [self.split_len1, self.split_len2], dim=1)
        if self.conditional:
            if x1.size(0) > c[0].size(0):
                gain_factor = x1.size(0) // c[0].size(0)
                repeat_tuple = tuple([gain_factor]+[1 for i in range(len(c[0].shape)-1)])
                c = torch.cat(c, 1).repeat(repeat_tuple)
                x1_c = torch.cat([x1, c], 1)
            elif x1.size(0) == c[0].size(0):
                x1_c = torch.cat([x1, *c], 1)
            else:
                raise NotImplementedError
        else:
            x1_c = x1

        t = self.subnet(x1_c)
        if rev:
            y2 = x2 - t
        else:
            y2 = x2 + t

        return (torch.cat((x1, y2), 1),), 0.

class ConditionalAffineTransform(_BaseCouplingBlock):
    '''Similar to the conditioning layers from SPADE (Park et al, 2019): Perform
    an affine transformation on the whole input, where the affine coefficients
    are predicted from only the condition.
    '''

    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor: Callable = None,
                 clamp: float = 2.,
                 clamp_activation: Union[str, Callable] = "ATAN", nn_settings=None):
        '''
        Additional args in docstring of base class.
        Args:
          subnet_constructor: function or class, with signature
            constructor(dims_in, dims_out).  The result should be a torch
            nn.Module, that takes dims_in input channels, and dims_out output
            channels. See tutorial for examples. One subnetwork will be
            initialized in the block.
          clamp: Soft clamping for the multiplicative component. The
            amplification or attenuation of each input dimension can be at most
            exp(±clamp).
          clamp_activation: Function to perform the clamping. String values
            "ATAN", "TANH", and "SIGMOID" are recognized, or a function of
            object can be passed. TANH behaves like the original realNVP paper.
            A custom function should take tensors and map -inf to -1 and +inf to +1.
        '''

        super().__init__(dims_in, dims_c, clamp, clamp_activation)

        if not self.conditional:
            raise ValueError("ConditionalAffineTransform must have a condition")
        
        nn_type = nn_settings['nn_type']
        if nn_type == 'SimpleConvNet':
            c_hidden_factor = nn_settings['c_hidden_factor']
            dim = nn_settings['dim']
            self.subnet = subnet_constructor(c_in=self.condition_length, \
                                        c_out = 2 * self.channels, c_hidden_factor=c_hidden_factor, dim=dim, resolution = dims_in[0][1:])
        elif nn_type == 'nnflowpp':
            coupling = 'Affine'
            in_channels = self.condition_length
            out_channels = 2 * self.channels
            num_channels_factor = nn_settings['num_channels_factor']
            num_blocks = nn_settings['num_blocks']
            drop_prob = nn_settings['drop_prob']
            use_attn = nn_settings['use_attn']
            #aux_channels = nn_settings['aux_channels']
            self.subnet = subnet_constructor(coupling, in_channels, out_channels, num_channels_factor, num_blocks, drop_prob, use_attn)


    def forward(self, x, c=[], rev=False, jac=True):
        if x[0].size(0) > c[0].size(0):
            gain_factor = x[0].size(0) // c[0].size(0)
            repeat_tuple = tuple([gain_factor]+[1 for i in range(len(c[0].shape)-1)])
            cond = torch.cat(c, 1).repeat(repeat_tuple)
        elif x[0].size(0) == c[0].size(0):
            cond = torch.cat(c, 1)
        else:
            raise NotImplementedError

        # notation:
        # x: inputs (i.e. 'x-side' when rev is False, 'z-side' when rev is True)
        # y: outputs (same scheme)
        # *_c: variables with condition appended
        # *1, *2: left half, right half
        # a: all affine coefficients
        # s, t: multiplicative and additive coefficients
        # j: log det Jacobian

        a = self.subnet(cond)
        s, t = a[:, :self.channels], a[:, self.channels:]
        s = self.clamp * self.f_clamp(s)
        j = torch.sum(s, dim=tuple(range(1, self.ndims + 1)))

        if rev:
            y = (x[0] - t) * torch.exp(-s)
            return (y,), -j
        else:
            y = torch.exp(s) * x[0] + t
            return (y,), j

class ConditionalAdditiveTransform(_BaseCouplingBlock):
    '''Similar to the conditioning layers from SPADE (Park et al, 2019): Perform
    an affine transformation on the whole input, where the affine coefficients
    are predicted from only the condition.
    '''

    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor: Callable = None,
                 clamp: float = 2.,
                 clamp_activation: Union[str, Callable] = "ATAN", nn_settings=None):
        '''
        Additional args in docstring of base class.
        Args:
          subnet_constructor: function or class, with signature
            constructor(dims_in, dims_out).  The result should be a torch
            nn.Module, that takes dims_in input channels, and dims_out output
            channels. See tutorial for examples. One subnetwork will be
            initialized in the block.
          clamp: Soft clamping for the multiplicative component. The
            amplification or attenuation of each input dimension can be at most
            exp(±clamp).
          clamp_activation: Function to perform the clamping. String values
            "ATAN", "TANH", and "SIGMOID" are recognized, or a function of
            object can be passed. TANH behaves like the original realNVP paper.
            A custom function should take tensors and map -inf to -1 and +inf to +1.
        '''

        super().__init__(dims_in, dims_c, clamp, clamp_activation)

        if not self.conditional:
            raise ValueError("ConditionalAffineTransform must have a condition")
        
        nn_type = nn_settings['nn_type']
        if nn_type == 'SimpleConvNet':
            c_hidden_factor = nn_settings['c_hidden_factor']
            dim = nn_settings['dim']
            self.subnet = subnet_constructor(c_in=self.condition_length, \
                                        c_out = self.channels, c_hidden_factor=c_hidden_factor, dim=dim, resolution = dims_in[0][1:])
        elif nn_type == 'nnflowpp':
            coupling = 'Affine'
            in_channels = self.condition_length
            out_channels = self.channels
            num_channels_factor = nn_settings['num_channels_factor']
            num_blocks = nn_settings['num_blocks']
            drop_prob = nn_settings['drop_prob']
            use_attn = nn_settings['use_attn']
            #aux_channels = nn_settings['aux_channels']
            self.subnet = subnet_constructor(coupling, in_channels, out_channels, num_channels_factor, num_blocks, drop_prob, use_attn)


    def forward(self, x, c=[], rev=False, jac=True):
        if x[0].size(0) > c[0].size(0):
            gain_factor = x[0].size(0) // c[0].size(0)
            repeat_tuple = tuple([gain_factor]+[1 for i in range(len(c[0].shape)-1)])
            cond = torch.cat(c, 1).repeat(repeat_tuple)
        elif x[0].size(0) == c[0].size(0):
            cond = torch.cat(c, 1)
        else:
            raise NotImplementedError

        t = self.subnet(cond)
        if rev:
            y = x[0] - t
            return (y,), 0.
        else:
            y = x[0] + t
            return (y,), 0.

"""---------------- deprecated code ----------------------"""

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
        #self.scaling_factor = nn.Parameter(torch.zeros(c_in)) #learnable scaling factor for each dimension

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
            
            if kwargs.get('invert') == True:
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
        #s_fac = self.scaling_factor.exp().view(infer_shape)
        #s = torch.tanh(s / s_fac) * s_fac
        s = 2*torch.tanh(s)

        # Mask outputs (only transform the second part)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        # Affine transformation
        if not reverse:
            z = torch.exp(s) * z + t
            #z = (z + t) * torch.exp(s)
            logdet += s.sum(dim=[i+1 for i in range(self.dim+1)]) #use self.dim+1 because we need to add over the channel dimension as well
        else:
            z = (z - t) * torch.exp(-s)
            #z = (z * torch.exp(-s)) - t
            logdet -= s.sum(dim=[i+1 for i in range(self.dim+1)])

        return z, logdet