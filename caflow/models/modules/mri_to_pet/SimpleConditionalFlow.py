#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 17:22:20 2021

@author: gbatz97
"""

import torch
import torch.nn as nn
from caflow.models.modules.blocks.ConditionalFlowBlock import g_I

class SimpleConditionalFlow(nn.Module):
    def __init__(self, channels, dim, resolution, scales, scale_depth, nn_settings):
        super(SimpleConditionalFlow, self).__init__()
        self.scale_flows = nn.ModuleList()

        #calculate the number of channels and resolution of L_N
        self.top_channels = 2**(dim-1)*channels #number of channels of the top latent
        self.top_resolution = [x//2 for x in resolution] #resolution of the top latent
        self.dim = dim
        self.scales = scales

        for scale in range(self.scales):
            g_I_channels = self.calculate_scale_channels(dim, scale, flow_type='g_I')
            g_I_resolution = self.calculate_scale_resolution(scale)
            scale_flow = g_I(channels=g_I_channels, dim=dim, \
                             resolution=g_I_resolution, depth=scale_depth, 
                             nn_settings=nn_settings)
            
            self.scale_flows.append(scale_flow)
        
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

    def calculate_scale_resolution(self, scale):
        return tuple([x//2**scale for x in self.top_resolution])

    def calculate_scale_channels(self, dim, scale, flow_type='g_I'):
        if scale < self.scales-1:
            return 2**((dim-1)*scale)*self.top_channels
        elif scale == self.scales-1: #last scale
            if flow_type=='g_I':
                return 2**((dim-1)*(scale-1))*2**dim*self.top_channels
            elif flow_type=='g_S':
                return 2**((dim-1)*scale)*self.top_channels
        
    
    def forward(self, L=[], z=[], D=[], logdet=0., reverse=False):
        if reverse:
            L_pred, logdet = self.decode(z=z, D=D, logdet=logdet)
            return L_pred, logdet
        else:
            z_enc, logprob, logdet = self.encode(L=L, D=D, logdet=logdet)
            return z_enc, logprob, logdet
    
    def encode(self, L, D, logdet):
        logprob = 0.
        z=[]

        for i in range(self.scales):
            h_split, logdet = self.scale_flows[i](L[i], D[i], logdet, reverse=False)
            logprob += self.prior.log_prob(h_split).sum(dim = [i+1 for i in range(self.dim+1)])
            z.append(h_split)
        
        return z, logprob, logdet
    
    def decode(self, z, D, logdet):
        L = []
        for i in range(self.scales):
            print(z[i].size(), D[i].size())
            h_pass, logdet = self.scale_flows[i](z[i], D[i], logdet, reverse=True)
            L.append(h_pass)
        return L, logdet