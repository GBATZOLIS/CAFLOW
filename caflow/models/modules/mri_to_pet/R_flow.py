#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 22:02:53 2021

@author: gbatz97
"""



from caflow.models.modules.blocks import AffineCouplingLayer
import torch.nn as nn


class flow_block(nn.Module):
    def __init__(self, shape):
        super(flow_block, self).__init__()
    
    def forward(self, h_split, h_pass, logdet=0., reverse=False)
        if reverse:
            if h_pass is not None:
                #concatenate h_split and h_pass
            else:
                #use only h_split for the reverse flow block
            
            #pass the tensor through the reverse flow block
            #return h_pass, logdet
        
        else:
            #pass h_pass through the forward flow
            #split the end tensor into h_pass and h_split
            #return h_split, h_pass, logdet




class R_flow(nn.Module):
    def __init__(self, shape, scales, opts):
        super(R_flow, self).__init__()

        self.scale_blocks = nn.ModuleList()

        for scale in range(1,scales+1):
            scale_shape = calculate_scale_shape(scale, original_shape)
            self.scale_blocks.append(flow_block(shape=scale_shape))

        # Create prior distribution for final latent space
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
        self.device = opts.device['R_flow']

    def calculate_scale_shape(scale, original_shape):
        #use this method to calculate the input shape of the flow block
        pass

    def forward(y=None, z=None, logdet=0., reverse=False):

        if reverse:
            assert z is not None
            y_dec, logdet = self.decode(z,logdet=logdet)
            return y_dec, logdet
        else:
            assert y is not None
            z_enc, logdet = self.encode(y, logdet=logdet)
            return z_enc, logdet
    
    def encode(y, logdet):

        h_pass = y
        z_enc = []
        for scale_block in self.scale_blocks:
            h_split, h_pass, logdet = scale_block(h_split=None, h_pass=h_pass, logdet=logdet, reverse=False)
            z_enc.append(h_split)

        return z_enc, logdet


    def decode(z, logdet):

        h_pass=None
        for h_split, scale_block in zip(reversed(z), reversed(self.scale_blocks)):
            h_pass, logdet = scale_block(h_split=h_split, h_pass=h_pass, logdet=logdet, reverse=True)
        y_dec = h_pass

        return y_dec, logdet


        

        

