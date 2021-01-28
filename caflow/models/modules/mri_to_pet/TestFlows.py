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

"""
# This file is used for testing of the implementation of the unconditional 
# and conditional flows under the mri_to_pet file
"""

#instantiate the unconditional flow
rflow = UnconditionalFlow(channels=1, dim=2, scales=5, scale_depth=2, network = GatedConvNet)
tflow = UnconditionalFlow(channels=1, dim=2, scales=5, scale_depth=2, network = GatedConvNet)


Y = torch.randn((3, 1, 512, 512), dtype=torch.float32)
print('y shape: ', Y.size())

I = torch.randn((3, 1, 512, 512), dtype=torch.float32)

print('Encoding Y and I with the forward pass...We get D and L.')
with torch.no_grad():
    D, logprior, logdet = rflow(y=Y)
    L, logprior, logdet = tflow(y=Y)

print('z_enc elements:')
for i, (D_i, L_i) in enumerate(zip(D,L)):
    print(i, D_i.size(), L_i.size())


condflow = SharedConditionalFlow(dim=2, scales=5, scale_depth=2, network=CondGatedConvNet)

#Use D, L to get the conditional enconding of L without the shortcut.
with torch.no_grad():
    z_no_short, logprob, logdet = condflow(L=L, z=[], D=D, reverse=False, shortcut=True)
    print('z_no_short: ', z_no_short)
    print('logprob.size() :', logprob.size())
    print('logdet.size() :', logdet.size())
    L_pred, logdet = condflow(L=[], z=z_no_short, D=D, reverse=True, shortcut=True)


for L_i, L_i_inv in zip(L, L_pred):
    r = torch.abs(L_i-L_i_inv)
    print('torch.sum(residual): ', torch.sum(r))
    print('torch.mean(residual): ', torch.mean(r))
 
    



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