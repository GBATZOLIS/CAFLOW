#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 00:58:42 2021

@author: gbatz97
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class Dequantisation(nn.Module):
    def __init__(self, dim=2, quants=256, alpha=1e-5):
        """
        Inputs:
            alpha - small constant that is used to scale the original input.
                    Prevents dealing with values very close to 0 and 1 when inverting the sigmoid
            quants - Number of possible discrete values (usually 256 for 8-bit image)
        """
        super().__init__()
        self.dim = dim
        self.quants = quants
        self.alpha = alpha
        
    def forward(self, z, ldj, reverse=False):
        if not reverse:
            z, ldj = self.dequant(z, ldj)
            z, ldj = self.sigmoid(z, ldj, reverse=True) # This is correct. you need the invert sigmoid after dequantisation.
        else:
            z, ldj = self.sigmoid(z, ldj, reverse=False)
            z = z * self.quants
            ldj += np.log(self.quants) * np.prod(z.shape[1:])
            z = torch.floor(z).clamp(min=0, max=self.quants-1)
        return z, ldj

    def sigmoid(self, z, ldj, reverse=False):
        # Applies an invertible sigmoid transformation
        if not reverse:
            ldj += (-z-2*F.softplus(-z)).sum(dim = [i+1 for i in range(self.dim+1)])
            z = torch.sigmoid(z)
        else:
            z = z * (1 - self.alpha) + 0.5 * self.alpha  # Scale to prevent boundaries 0 and 1
            ldj += np.log(1 - self.alpha) * np.prod(z.shape[1:])
            
            ldj += (-torch.log(z) - torch.log(1-z)).sum(dim = [i+1 for i in range(self.dim+1)])
            z = torch.log(z) - torch.log(1-z)
            
        return z, ldj

    def dequant(self, z, ldj):
        # Transform discrete values to continuous volumes
        z = z + torch.rand_like(z) #add uniform noise U(0,1)
        z = z / self.quants 
        ldj += -1 * np.log(self.quants) * np.prod(z.shape[1:])
        return z, ldj



class VariationalDequantization(Dequantisation):

    def __init__(self, var_flows, dim=2, quants=256, alpha=1e-5):
        """
        Inputs:
            var_flows - A list of flow transformations to use for modeling q(u|x)
            alpha - Small constant, see Dequantization for details
        """
        super().__init__(dim, quants, alpha)
        self.flows = nn.ModuleList(var_flows)

    def dequant(self, z, ldj):
        img = (z / (self.quants-1)) * 2 - 1 # We condition the flows on x, i.e. the original image

        # Prior of u is a uniform distribution as before
        # As most flow transformations are defined on [-infinity,+infinity], we apply an inverse sigmoid first.
        deq_noise = torch.rand_like(z)
        deq_noise, ldj = self.sigmoid(deq_noise, ldj, reverse=True)
        for flow in self.flows:
            deq_noise, ldj = flow(deq_noise, ldj, reverse=False, orig_img=img)
        deq_noise, ldj = self.sigmoid(deq_noise, ldj, reverse=False)

        # After the flows, apply u as in standard dequantization
        z = (z + deq_noise) / self.quants
        ldj -= np.log(self.quants) * np.prod(z.shape[1:])
        return z, ldj

