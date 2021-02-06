#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 11:54:13 2021

@author: gbatz97
"""

from caflow.models.modules.mri_to_pet.UnconditionalFlow import UnconditionalFlow
from caflow.models.modules.mri_to_pet.SharedConditionalFlow import SharedConditionalFlow

from caflow.models.modules.networks.GatedConvNet import GatedConvNet
from caflow.models.modules.networks.CondGatedConvNet import CondGatedConvNet

import torch

import time

"""
# This file is used for testing of the implementation of the unconditional 
# and conditional flows under the mri_to_pet file
"""

def test_equality(z_short, z_normal):
    """z_normal"""
    # z = [ [z_(n-1)^(n-1), z_(n-2)^(n-1), z_(n-3)^(n-1), z_(n-4)^(n-1), ..., z_1^(n-1), z_0^(n-1)],
    #       [z_(n-2)^(n-2), z_(n-3)^(n-2), z_(n-4)^(n-2), ..., z_1^(n-2), z_0^(n-2)],
    #       [z_(n-3)^(n-3), z_(n-4)^(n-3) ,..., z_1^(n-3), z_0^(n-3)],
    #       ...,
    #       [z_2^2,         z_1^2, z_0^2],
    #       [z_1^1,         z_0^1], 
    #       [z_0^0] 
    #     ]

    """z_short"""
    # z = [z_I, z_S]
    # z_I = [ z_(n-1)^(n-1), z_(n-2)^(n-2), ..., z_2^2, z_1^1, z_0^0 ]
    # z_S = [ z_(n-2)^(n-1),      
    #         z_(n-3)^(n-1)||z_(n-3)^(n-2),       
    #         z_(n-4)^(n-1)||z_(n-4)^(n-2)||z_(n-4)^(n-3), 
    #            ...,              
    #         z_1^(n-1)||z_1^(n-2)|| ... ||z_1^2,       
    #         z_0^(n-1)||z_0^(n-2)|| ... ||z_0^2||z_0^1
    #        ]
    # concatenation || takes place in the zero dimension (dim=0)

    # way to test equality
    # Create the z_short latent vector from the z_normal latent vector and test equality
    # We can also do the opposite because this is a helpful functionality

    def convert_shortcut_to_normal(z_short):
        z_I = z_short[0]
        z_S = z_short[1]

        n = len(z_I)
        z = []
        for i in range(n):
            iflow = []
            iflow.append(z_short[0][i])
            z.append(iflow)
        
        for i in range(1, n):
            if i==1:
                z[0].append(z_S[i-1])
            else:
                batch = z_S[i-1].size(0)//i
                for j in range(i):
                    z[j].append(z_S[i-1][batch*j:batch*(j+1)])
        
        return z

    def convert_normal_to_shortcut(z_normal):
        n = len(z_normal)

        z_I = []
        for i in range(n):
            z_I.append(z_normal[i][0])
        
        z_S = []
        for i in range(1, n):
            if i==1:
                z_S.append(z_normal[0][1])
            else:
                concat_tensor = torch.cat([z_normal[j][i-j] for j in range(i)], dim=0)
                z_S.append(concat_tensor)

        z_short_converted = [z_I, z_S]
        return z_short_converted

    print('--------------CONVERT NORMAL TO SHORTCUT----------------')
    z_short_converted = convert_normal_to_shortcut(z_normal)
    for i in range(len(z_short)):
        if i==0:
            print('-----z_I comparison-----')
        elif i==1:
            print('-----z_S comparison-----')

        for j, (converted, real) in enumerate(zip (z_short_converted[i], z_short[i])):
            print('Element %d: summed absolute difference  :  %.16f' %(j, torch.sum(torch.abs(converted - real))))
            print(converted.size(), real.size())
    
    print('--------------CONVERT SHORTCUT TO NORMAL ----------------')
    z_normal_converted = convert_shortcut_to_normal(z_short)
    for i in range(len(z_normal)):
        print('-------%d flow-------' % i)
        for j, (converted, real) in enumerate(zip(z_normal_converted[i], z_normal[i])):
            print('Element %d: summed absolute difference  :  %.16f' %(j, torch.sum(torch.abs(converted - real))))
            #print(converted.size(), real.size())






#instantiate the unconditional flow
rflow = UnconditionalFlow(channels=3, dim=2, scales=6, scale_depth=1)
tflow = UnconditionalFlow(channels=3, dim=2, scales=6, scale_depth=1)


Y = torch.randn((3, 3, 256, 256), dtype=torch.float32)
print('y shape: ', Y.size())

I = torch.randn((3, 3, 256, 256), dtype=torch.float32)

print('Encoding Y and I with the forward pass...We get D and L.')
with torch.no_grad():
    D, logprior, logdet = rflow(y=Y)
    L, logprior, logdet = tflow(y=I)

print('z_enc elements:')
for i, (D_i, L_i) in enumerate(zip(D,L)):
    print(i, D_i.size(), L_i.size())


condflow = SharedConditionalFlow(channels=3, dim=2, scales=6, scale_depth=1)



#Use D, L to get the conditional enconding of L without the shortcut.
with torch.no_grad():
    start = time.time()
    z_normal, logprob, logdet = condflow(L=L, z=[], D=D, reverse=False, shortcut=False)
    L_pred_normal, logdet = condflow(L=[], z=z_normal, D=D, reverse=True, shortcut=False)
    end = time.time()
    normal_time = end - start
    
    
    start = time.time()
    z_short, logprob, logdet = condflow(L=L, z=[], D=D, reverse=False, shortcut=True)
    L_pred_short, logdet = condflow(L=[], z=z_short, D=D, reverse=True, shortcut=True)
    end = time.time()
    shortcut_time = end - start

print('normal time: ', normal_time)
print('shortcut time: ', shortcut_time)

test_equality(z_short, z_normal) #passed
    



for L_i, L_pred_normal_i, L_pred_short_i in zip(L, L_pred_normal, L_pred_short):
    r_normal = torch.abs(L_i - L_pred_normal_i)
    r_shortcut = torch.abs(L_i - L_pred_short_i)
    r_shortcut_normal = torch.abs(L_pred_short_i - L_pred_normal_i)
    
    print('---------------------------')
    print('torch.sum(r_normal): ', torch.sum(r_normal))
    print('torch.mean(r_normal): ', torch.mean(r_normal))
    print('torch.max(r_normal): ', torch.max(r_normal))
    print('------')
    print('torch.sum(r_shortcut): ', torch.sum(r_shortcut))
    print('torch.mean(r_shortcut): ', torch.mean(r_shortcut))
    print('torch.max(r_shortcut): ', torch.max(r_shortcut))
    print('------')
    print('torch.sum(r_shortcut_normal): ', torch.sum(r_shortcut_normal))
    print('torch.mean(r_shortcut_normal): ', torch.mean(r_shortcut_normal))
    print('torch.max(r_shortcut_normal): ', torch.max(r_shortcut_normal))