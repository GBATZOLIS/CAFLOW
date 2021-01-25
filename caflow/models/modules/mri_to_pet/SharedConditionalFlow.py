#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 22:02:53 2021

@author: gbatz97
"""



from caflow.models.modules.blocks import *
import torch.nn as nn



class SharedConditionalFlow(nn.Module):
    def __init__(self, channels, dim, scales, opts):
        super(SharedConditionalFlow, self).__init__()

        self.g_I_cond_flows = nn.ModuleList()
        self.g_S_cond_flows = nn.ModuleList()
        
        self.channels = channels #initial number of channels of the image (3 for rgb images)
        self.dim = dim
        self.scales = scales
        
        for scale in range(1, self.scales+1):
            scale_channels = self.calculate_scale_channels(dim, scale)
            print(scale_channels)
            
            self.g_I_cond_flows.append(g_I_cond_flow(scale_channels))
            
            if scale > 1:#There is no shared flow in the first level
                self.g_S_cond_flows.append(g_S_cond_flow(scale_channels))
        
        #self.device = opts.device['Cond_flow'] #we will take care of that later
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

    def calculate_scale_channels(self, dim, scale):
        if scale < self.scales:
            return 2**((dim-1)*scale)*self.channels
        elif scale == self.scales: #last scale
            return 2**((dim-1)*(scale-1))*2**dim*self.channels
        
    def forward(self, L=[], z=[], D=[], logdet=0., reverse=False, shortcut=False):
        if reverse:
            L_pred, logdet = self.decode(z, D, shortcut)
        else:
            logprob, logdet = self.encode(L, D, shortcut)

    def encode(self, L, D, logdet, shortcut=False):
        z=[]
        if not shortcut:
            """checked"""
            logprob = 0.
            for i in range(self.scales):
                z[i]=[]
                
                if i==self.scales-1:
                    h_split, logdet = self.g_I_cond_flows[i](L[i], D[i], logdet, reverse=False)
                    logprob += self.prior.log_prob(h_split).sum(dim = [i+1 for i in range(self.dim+1)])
                    z[i].append(h_split)
                    continue

                h_pass, logdet = self.g_I_cond_flows[i](L[i], D[i],logdet,reverse=False)
                h_split, h_pass = h_pass.chunk(2, dim=1)
                logprob += self.prior.log_prob(h_split).sum(dim = [i+1 for i in range(self.dim+1)])
                z[i].append(h_split)

                for j in range(i, self.scales-1):
                    if j==self.scales-2:
                        h_split, logdet = self.g_S_cond_flows[j](h_pass, L[j+1], D[j+1], logdet, reverse=False)
                        logprob += self.prior.log_prob(h_split).sum(dim = [i+1 for i in range(self.dim+1)])
                        z[i].append(h_split)
                    
                    h_pass, logdet = self.g_S_cond_flows[j](h_pass, L[j+1], D[j+1], 
                                                            logdet, reverse=False)
                    h_split, h_pass = h_pass.chunk(2, dim=1)
                    logprob += self.prior.log_prob(h_split).sum(dim = [i+1 for i in range(self.dim+1)])
                    z[i].append(h_split)

            return logprob, logdet
        else:
            logprob = 0.
            previous_scale_I_activation = None
            previous_scale_S_activations = None
            for i in range(self.scales):
                if i == self.scales-1:
                    h_split, logdet = self.g_I_cond_flows[i](L[i], D[i], logdet, 
                                                        reverse=False, split=False)
                    logprob += self.logprob(h_split)
                    continue

                h_split, h_pass, logdet = self.g_I_cond_flows[i](L[i], D[i], logdet, 
                                                            reverse=False, split=True)
                logprob += self.logprob(h_split)
                previous_scale_I_activation = h_pass

                if previous_scale_S_activations==None:
                    h_split, h_pass, logdet = self.g_S_cond_flows[i](previous_scale_I_activation, 
                                                                L[i+1], D[i+1], logdet, 
                                                                reverse=False, split=True)
                    logprob += self.logprob(h_split)
                    previous_scale_S_activations = h_pass
                    continue
                
                concat_scale_activations = torch.cat([previous_scale_I_activation, 
                                                    previous_scale_S_activations], dim=0)
                
                if i == self.scales-2:
                    h_split, logdet = self.g_S_cond_flows[i](concat_scale_activations, 
                                                        L[i+1], D[i+1], logdet, 
                                                        reverse=False, split=False)
                    logprob += self.logprob(h_split)
                    continue

                h_split, h_pass, logdet = self.g_S_cond_flows[i](concat_scale_activations, 
                                                            L[i+1], D[i+1], logdet, 
                                                            reverse=False, split=True)
                logprob += self.logprob(h_split)
                previous_scale_S_activations = h_pass

            return logprob, logdet
    
    def decode(self, z, D, logdet, shortcut=False):
        D=D[::-1]
        g_S_cond_flows = self.g_S_cond_flows[::-1]
        g_I_cond_flows = self.g_I_cond_flows[::-1]
        
        L = []
        
        if not shortcut:
            z=z[::-1]
            #z is a list of the latent tensors of each flow
            #[[z_0],[z_0,z_1],...,[z_0,...z_{n-1}]]
            for i in range(self.scales):
                if i==0:
                    h_pass, logdet = g_I_cond_flows[i](z[i][0], D[i], logdet, reverse=True)
                    L.append(h_pass)
                    continue
                else:
                    for j in range(i):
                        if j==0:
                            h_pass, logdet = g_S_cond_flows[j](z[i][j], L[j], D[j], logdet, reverse=True)
                        else:
                            concat_pass = torch.cat([z[i][j], h_pass], dim=1)
                            h_pass, logdet = g_S_cond_flows[j](concat_pass, L[j], D[j], logdet, reverse=True)

                    concat_pass = torch.cat([z[i][i], h_pass], dim=1)
                    h_pass, logdet = g_I_cond_flows[i](concat_pass, D[i], logdet, reverse=True)
                    L.append(h_pass)
            
            L.inverse()
            return L, logdet

        else:
            z=[z[0][::-1], z[1][::-1]]
            #z is a list of two lists of latent tensors
            #the first list contains the I latent tensors and the second contains the S latent tensors
            #len_first = n, len_second = n-1
            #z = [[z_0_i,...,z_{n-1}_i], [z_0_i,...,z_{n-2}_i]]
            last_S_h_pass = None
            for i in range(self.scales):
                if i==0:
                    h_pass, logdet = g_I_cond_flows[i](z[0][i], D[i], logdet, reverse=True)
                    L.append(h_pass)
                    h_pass, logdet = g_S_cond_flows[i](z[1][i], L[-1], D[i], logdet, reverse=True)
                    last_S_h_pass = h_pass

                elif i<self.scales-1:
                    concat_pass = torch.cat([z[0][i], last_S_h_pass], dim=1)
                    h_pass, logdet = g_I_cond_flows[i](concat_pass, D[i], logdet, reverse=True)
                    L.append(h_pass)

                    concat_pass = torch.cat([z[1][i], last_S_h_pass], dim=1)
                    h_pass, logdet = g_S_cond_flows[i](concat_pass, L[-1], D[i], logdet, reverse=True)
                    last_S_h_pass = h_pass

                elif i==self.scales-1:
                    concat_pass = torch.cat([z[0][i], last_S_h_pass], dim=1)
                    h_pass, logdet = g_I_cond_flows[i](concat_pass, D[i], logdet, reverse=True)
                    L.append(h_pass)

            L.inverse()
            return L, logdet


