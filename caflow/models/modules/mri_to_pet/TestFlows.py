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

#instantiate the unconditional flow
rflow = UnconditionalFlow(channels=3, dim=2, scales=3, scale_depth=2)
tflow = UnconditionalFlow(channels=3, dim=2, scales=3, scale_depth=2)


Y = torch.randn((2, 3, 64, 64), dtype=torch.float32)
print('y shape: ', Y.size())

I = torch.randn((2, 3, 64, 64), dtype=torch.float32)

print('Encoding Y and I with the forward pass...We get D and L.')
with torch.no_grad():
    D, logprior, logdet = rflow(y=Y)
    L, logprior, logdet = tflow(y=I)

print('z_enc elements:')
for i, (D_i, L_i) in enumerate(zip(D,L)):
    print(i, D_i.size(), L_i.size())


condflow = SharedConditionalFlow(channels=3, dim=2, scales=3, scale_depth=2)



#Use D, L to get the conditional enconding of L without the shortcut.
with torch.no_grad():
    start = time.time()
    z_normal, logprob, logdet = condflow(L=L, z=[], D=D, reverse=False, shortcut=False)
    print(logprob)
    print(logdet)
    L_pred_normal, logdet = condflow(L=[], z=z_normal, D=D, reverse=True, shortcut=False)
    print(logdet)
    end = time.time()
    normal_time = end - start
    
    
    start = time.time()
    z_short, logprob, logdet = condflow(L=L, z=[], D=D, reverse=False, shortcut=True)
    print(logprob)
    print(logdet)
    L_pred_short, logdet = condflow(L=[], z=z_short, D=D, reverse=True, shortcut=True)
    print(logdet)
    end = time.time()
    shortcut_time = end - start
    


"""
for L_i, L_pred_normal_i, L_pred_short_i in zip(L, L_pred_normal, L_pred_short):
    r_normal = torch.abs(L_i - L_pred_normal_i)
    r_shortcut = torch.abs(L_i - L_pred_short_i)
    r_shortcut_normal = torch.abs(L_pred_short_i - L_pred_normal_i)
    
    print('---------------------------')
    print('torch.sum(r_normal): ', torch.sum(r_normal))
    print('torch.mean(r_normal): ', torch.mean(r_normal))
    print('------')
    print('torch.sum(r_shortcut): ', torch.sum(r_shortcut))
    print('torch.mean(r_shortcut): ', torch.mean(r_shortcut))
    print('------')
    print('torch.sum(r_shortcut_normal): ', torch.sum(r_shortcut_normal))
    print('torch.mean(r_shortcut_normal): ', torch.mean(r_shortcut_normal))

"""



print('normal time: ', normal_time)
print('shortcut time: ', shortcut_time)
    
 
    



"""
condflow = SharedConditionalFlow(dim=3, scales=4, scale_depth=2, network=CondGatedConvNet)

#Use D, L to get the conditional enconding of L without the shortcut.
z_no_short, logprob, logdet = condflow(L=L, z=[], D=D, reverse=False, shortcut=False)

#Use the enconding to retriece the original L
L_pred, logdet = condflow(L=[], z=z_no_short, D=D, reverse=True, shortcut=False)

r = torch.abs(L_pred[0]-L[0])
print('sum(|y-y_dec|)',torch.sum(r))
print('mean(|y-y_dec|):',torch.mean(r))
"""