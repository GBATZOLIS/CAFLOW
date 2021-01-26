#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 22:02:53 2021

@author: gbatz97
"""



from caflow.models.modules.blocks import *
import torch.nn as nn
import torch


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
            z_enc, logprob, logdet = self.encode(L, D, shortcut)

    def encode(self, L, D, logdet, shortcut=False):
        
        if not shortcut:
            """done"""
            logprob = 0.
            z=[]
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

            return z, logprob, logdet
        else:
            """done"""
            logprob = 0.
            previous_scale_I_activation = None
            previous_scale_S_activations = None
            z_I = []
            z_S = []
            for i in range(self.scales):
                # calculate all the activations at the i^th level. Calculations flow from right to left in scale-wise manner
                # 1.) We calculate the activation of g_I_i at this scale/level(i).
                # 2.) We concatenate all the activations of the previous scale/level including the g_I activation
                #     and we pass them through the g_S network of the i^th scale/level.
                
                #1.) calculate g_I_i activation
                if i == self.scales-1:
                    #last g_I activation is not split.
                    h_split, logdet = self.g_I_cond_flows[i](L[i], D[i], logdet, reverse=False)
                    logprob += self.logprob(h_split)
                    z_I.append(h_split)
                    continue
                
                
                h_pass, logdet = self.g_I_cond_flows[i](L[i], D[i], logdet, reverse=False)
                h_split, h_pass = h_pass.chunk(2, dim=1)
                logprob += self.logprob(h_split)
                z_I.append(h_split)
                # we need to store h_pass because it will be concatenated with the rest of the activations of the same level,
                # so that they can be concatenated and passed through the next g_S network altogether.
                previous_scale_I_activation = h_pass 
                
                #no concatenation is needed in the first g_S flow.
                if previous_scale_S_activations==None:
                    h_pass, logdet = self.g_S_cond_flows[i](previous_scale_I_activation, 
                                                            L[i+1], D[i+1], logdet, reverse=False)
                    h_split, h_pass = h_pass.chunk(2, dim=1)
                    logprob += self.logprob(h_split)
                    z_S.append(h_split)
                    previous_scale_S_activations = h_pass
                    continue
                
                #concatenate all the activations coming from the previous level/scale
                concat_scale_activations = torch.cat([previous_scale_S_activations, 
                                                      previous_scale_I_activation], dim=0)
                
                if i == self.scales-2:
                    #last g_S activation is not split.
                    h_split, logdet = self.g_S_cond_flows[i](concat_scale_activations, 
                                                             L[i+1], D[i+1], logdet, reverse=False)
                    logprob += self.logprob(h_split)
                    z_S.append(h_split)
                    continue

                h_pass, logdet = self.g_S_cond_flows[i](concat_scale_activations, 
                                                        L[i+1], D[i+1], logdet, reverse=False)
                h_split, h_pass = h_pass.chunk(2, dim=1)
                logprob += self.logprob(h_split)
                z_S.append(h_split)
                previous_scale_S_activations = h_pass

            z_enc = [z_I, z_S]
            return z_enc, logprob, logdet
    
    def decode(self, z, D, logdet, shortcut=False, xtreme_shortcut=False):
        #D = [D_{n-1}, D_{n-2}, ..., D_2, D_1, D_0]
        #self.g_I_cond_flows = [          g_S_{n-2}, g_S_{n-3}, ..., g_S_2, g_S_1, g_S_0]
        #self.g_I_cond_flows = [g_I_{n-1},g_I_{n-2}, g_I_{n-3}, ..., g_I_2, g_I_1, g_I_0]

        L = []
        
        # There are two types of shortcut in the decoding process.
        # There is the shortcut that just reverses the shortcut version of encoding
        # There is also the extreme shortcut that computes each g_S^(-1) activation exactly once.
        # The latter shortcut can be used for fast sampling at the expense of less diversity in the output.
            
        if not shortcut:
            """done"""
            #z is a list of the latent tensors of each flow
            #z = [[z_{n-1}, ..., z_0],[z_{n-2}, ..., z_0],...,[z_2, z_1, z_0],[z_1, z_0], [z_0]]
            n=self.scales
            for i in range(n):
                if i==0:
                    h_pass, logdet = self.g_I_cond_flows[-1](z[-1][0], D[-1], logdet, reverse=True)
                    L.append(h_pass) #L_0
                    continue
                else:
                    for j in range(i):
                        if j==0:
                            h_pass, logdet = self.g_S_cond_flows[-1](z[-1-i][-1], L[0], D[-1], logdet, reverse=True)
                        
                        else:
                            concat_pass = torch.cat([z[-1-i][-1-j], h_pass], dim=1)
                            h_pass, logdet = self.g_S_cond_flows[-1-j](concat_pass, L[j], D[-1-j], logdet, reverse=True)

                    concat_pass = torch.cat([z[-1-i][0], h_pass], dim=1)
                    h_pass, logdet = self.g_I_cond_flows[-1-i](concat_pass, D[-1-i], logdet, reverse=True)
                    L.append(h_pass)

            #L=[L_0, L_1,...,L_{n-1}]
            L.inverse() #we need to invert L because the subsequent unconditinal flow exptects the inverted order
            return L, logdet
        
        elif shortcut and not xtreme_shortcut:
            """done"""
            #D = [D_(n-1), D_(n-2), ..., D_2, D_1, D_0]
            z_I=z[0] #[z_(n-1)^(n-1), z_(n-2)^(n-2), ..., z_2^2, z_1^1, z_0^0]
            z_S=z[1] #[z_(n-2)^(n-1),      z_(n-3)^(n-1)||z_(n-3)^(n-2),       z_(n-4)^(n-1)||z_(n-4)^(n-2)||z_(n-4)^(n-3), 
                     # ...,              z_1^(n-1)||z_1^(n-2)|| ... ||z_1^2,       z_0^(n-1)||z_0^(n-2)|| ... ||z_0^1     ]
            
            n=self.scales:
            for i in range(n):
                if i==0:
                    h_pass, logdet = self.g_I_cond_flows[-1](z_I[-1], D[-1], logdet, reverse=True)
                    L.append(h_pass) #L_0
                    h_pass, logdet = self.g_S_cond_flows[-1](z_S[-1], L[0], D[-1], logdet, reverse=True)
                    continue
                
                elif i <= n-2:
                    batch_size = h_pass.shape[0]//(n-i)
                    h_pass_I = h_pass[(n-i-1)*batch_size:]
                    h_pass_S = h_pass[:(n-i-1)*batch_size]

                    concat_pass_I = torch.cat([z_I[-1-i], h_pass_I], dim=1)
                    h_pass, logdet = self.g_I_cond_flows[-1-i](concat_pass_I, D[-1-i], logdet, reverse=True)
                    L.append(h_pass) #L_i

                    concat_pass_S = torch.cat([z_S[-1-i], h_pass_S], dim=1)
                    h_pass, logdet = self.g_S_cond_flows[-1-i](concat_pass_S, L[i], D[-1-i], logdet, reverse=True)
                
                elif i == n-1:
                    h_pass_I = h_pass
                    concat_pass_I = torch.cat([z_I[-1-i], h_pass_I], dim=1)
                    h_pass, logdet = self.g_I_cond_flows[-1-i](concat_pass_I, D[-1-i], logdet, reverse=True)
                    L.append(h_pass) #L_(n-1)
            
            # L=[L_0, L_1,...,L_(n-1)]
            L.inverse() 
            # L=[L_(n-1), ..., L_1, L_0]
            return L, logdet
        
        elif shortcut and xtreme_shortcut:
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


