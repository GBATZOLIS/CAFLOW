#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 11:50:34 2021

@author: gbatz97
"""


import torch

x = torch.randn((1, 1, 2, 2), dtype=torch.float32)
x = x.repeat((3,1,1,1))
print(x.size())

print(x[0])
print(x[1])
print(x[2])
print(type(x.size(0)))