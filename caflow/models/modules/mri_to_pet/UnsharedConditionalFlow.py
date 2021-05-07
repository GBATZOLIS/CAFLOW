#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 17:22:20 2021

@author: gbatz97
"""

import torch
import torch.nn as nn
from caflow.models.modules.blocks.ConditionalFlowBlock import g_S, g_I

class UnsharedConditionalFlow(nn.Module):
    def __init__(self, channels, dim, resolution, scales, scale_depth, nn_settings):
        super(UnsharedConditionalFlow, self).__init__()
        self.scale_flows = nn.ModuleList()

        #calculate the number of channels and resolution of L_N
        self.top_channels = 2**(dim-1)*channels #number of channels of the top latent
        self.top_resolution = [x//2 for x in resolution] #resolution of the top latent
        self.dim = dim
        self.scales = scales

        for scale in range(self.scales):
            scale_flow = nn.ModuleList()

            g_I_channels = self.calculate_scale_channels(dim, scale, flow_type='g_I')
            g_I_resolution = self.calculate_scale_resolution(scale)
            scale_flow.append(g_I(channels=g_I_channels, dim=dim, \
                                  resolution=g_I_resolution, depth=scale_depth, nn_settings=nn_settings))
            
            if scale < self.scales - 1:
                for continued_scale in range(scale+1, self.scales):
                    g_S_channels = self.calculate_scale_channels(dim, continued_scale, flow_type='g_S')
                    g_S_resolution = self.calculate_scale_resolution(continued_scale)
                    last_scale = False if continued_scale < (self.scales-1) else True
                    scale_flow.append(g_S(channels=g_S_channels, dim=dim, resolution=g_S_resolution,
                                          depth=scale_depth, nn_settings=nn_settings, last_scale=last_scale))
            
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
            z_horizontal=[]                
            if i==self.scales-1:
                h_split, logdet = self.scale_flows[i][0](L[i], D[i], logdet, reverse=False)
                logprob += self.prior.log_prob(h_split).sum(dim = [i+1 for i in range(self.dim+1)])
                z_horizontal.append(h_split)
            else:
                h_pass, logdet = self.scale_flows[i][0](L[i], D[i], logdet, reverse=False)
                h_split, h_pass = h_pass.chunk(2, dim=1)
                logprob += self.prior.log_prob(h_split).sum(dim = [i+1 for i in range(self.dim+1)])
                z_horizontal.append(h_split)
                for j in range(i+1, self.scales):
                    if j==self.scales-1:
                        h_split, logdet = self.scale_flows[i][j-i](h_pass, L[j], D[j], logdet, reverse=False)
                        logprob += self.prior.log_prob(h_split).sum(dim = [i+1 for i in range(self.dim+1)])
                        z_horizontal.append(h_split)
                    else:
                        h_pass, logdet = self.scale_flows[i][j-i](h_pass, L[j], D[j], logdet, reverse=False)
                        h_split, h_pass = h_pass.chunk(2, dim=1)
                        logprob += self.prior.log_prob(h_split).sum(dim = [i+1 for i in range(self.dim+1)])
                        z_horizontal.append(h_split)
            
            z.append(z_horizontal)
            
        #return False, logprob, logdet
        return z, logprob, logdet
        
    
    def decode(self, z, D, logdet):
        #D = [D_{n-1}, D_{n-2}, ..., D_2, D_1, D_0]

        #z is a list of the latent tensors of each flow
        #z = [ [z_(n-1)^(n-1), z_(n-2)^(n-1), z_(n-3)^(n-1), z_(n-4)^(n-1), ..., z_1^(n-1), z_0^(n-1)],
        #      [z_(n-2)^(n-2), z_(n-3)^(n-2), z_(n-4)^(n-2), ..., z_1^(n-2), z_0^(n-2)],
        #      [z_(n-3)^(n-3), z_(n-4)^(n-3) ,..., z_1^(n-3), z_0^(n-3)],
        #      ...,
        #      [z_2^2,         z_1^2, z_0^2],
        #      [z_1^1,         z_0^1], 
        #      [z_0^0] 
        #    ]

        L = []
        n=self.scales
        for i in range(n):
            if i==0:
                h_pass, logdet = self.scale_flows[-1][0](z[-1][0], D[-1], logdet, reverse=True)
                L.append(h_pass) #L_0
                continue
            else:
                for j in range(i):
                    if j==0:
                        h_pass, logdet = self.scale_flows[-1-i][-1](z[-1-i][-1], L[0], D[-1], logdet, reverse=True)
                    else:
                        concat_pass = torch.cat([z[-1-i][-1-j], h_pass], dim=1)
                        h_pass, logdet = self.scale_flows[-1-i][-1-j](concat_pass, L[j], D[-1-j], logdet, reverse=True)

                concat_pass = torch.cat([z[-1-i][0], h_pass], dim=1)
                h_pass, logdet = self.scale_flows[-1-i][0](concat_pass, D[-1-i], logdet, reverse=True)
                L.append(h_pass)

        #L=[L_0, L_1,...,L_{n-1}]
        L=L[::-1] #we need to invert L because the subsequent unconditinal flow expects the inverted order
        return L, logdet
