#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 17:29:09 2021

@author: gbatz97
"""


from caflow.models.modules.mri_to_pet.UnconditionalFlow import UnconditionalFlow
from caflow.models.modules.mri_to_pet.SharedConditionalFlow import SharedConditionalFlow
import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.optim as optim

class CAFlow(pl.LightningModule):
    def __init__(self, opts):
        super(CAFlow, self).__init__()
        self.train_shortcut = opts.train_shortcut
        self.dim = opts.data_dim
        self.scales = opts.model_scales
        self.channels = opts.data_channels

        self.model = nn.ModuleDict()
        self.model['rflow'] = UnconditionalFlow(channels=opts.data_channels, dim=opts.data_dim, scales=opts.model_scales, 
                                  scale_depth=opts.model_scale_depth)
        self.model['tflow'] = UnconditionalFlow(channels=opts.data_channels, dim=opts.data_dim, scales=opts.model_scales, 
                                  scale_depth=opts.model_scale_depth)
        self.model['condflow'] = SharedConditionalFlow(channels=opts.data_channels, dim=opts.data_dim, scales=opts.model_scales, scale_depth=opts.model_scale_depth)
        
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

    def forward(self, Y, I, shortcut=False):
        return -1*torch.mean(self.logjoint(Y, I, shortcut)) #negative average log joint

    def logjoint(self, Y, I, shortcut=False):
        D, rlogprior, rlogdet = self.model['rflow'](y=Y)
        L, _, tlogdet = self.model['tflow'](y=I)
        Z_cond, condlogprior, condlogdet = self.model['condflow'](L=L, z=[], D=D, reverse=False, shortcut=shortcut)

        logjoint = rlogprior + rlogdet + tlogdet + condlogprior + condlogdet
        return logjoint
    
    def training_step(self, batch, batch_idx):
        Y, I = batch
        neg_avg_logjoint = self(Y, I, shortcut = self.train_shortcut)
        loss = neg_avg_logjoint
        
        #logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        Y, I = batch
        neg_avg_logjoint = self(Y, I, shortcut = self.train_shortcut)
        loss = neg_avg_logjoint
        self.log('val_loss', loss)
    
    def test_step(self, batch, batch_idx):
        Y, I = batch
        neg_avg_logjoint = self(Y, I, shortcut = self.train_shortcut)
        loss = neg_avg_logjoint
        self.log('test_loss', loss)
        
        
    def configure_optimizers(self,):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]
    
    @torch.no_grad()
    def sample(self, Y, Z_cond=None, shortcut=True):
        D, _, _ =self.model['rflow'](y=Y)

        if shortcut:
            #D = [D_(n-1), D_(n-2), ..., D_2, D_1, D_0]
            #z_I=z[0] #[z_(n-1)^(n-1), z_(n-2)^(n-2), ..., z_2^2, z_1^1, z_0^0]
            #z_S=z[1] #[z_(n-2)^(n-1),      z_(n-3)^(n-1)||z_(n-3)^(n-2),       z_(n-4)^(n-1)||z_(n-4)^(n-2)||z_(n-4)^(n-3), 
                        # ...,              z_1^(n-1)||z_1^(n-2)|| ... ||z_1^2,       z_0^(n-1)||z_0^(n-2)|| ... ||z_0^1     ]
            
            #start_shape = D[0].shape #(batchsize, 2*(dim-1)*self.channels, init_res/2, init_res/2, init_res/2)
            batch_size = D[0].shape[0]
            init_res = D[0].shape[2:]
            init_channels = D[0].shape[1]

            z_I = []
            z_S = []
            for scale in range(self.scales):
                if scale < self.scales-1:
                    right_shape = (batch_size, 2**((self.dim-1)*scale)*init_channels)+tuple([x//2**scale for x in init_res])
                    crop_left_shape = (batch_size, 2**(self.dim-1)*right_shape[1],)+tuple([x//2**scale for x in right_shape[2:]])
                    if scale == 0:
                        z_I.append(self.prior.sample(sample_shape=crop_left_shape))
                    else:
                        z_I.append(self.prior.sample(sample_shape=crop_left_shape))
                        concat_z_S = self.prior.sample(sample_shape=(scale*crop_left_shape[0])+crop_left_shape[1:])
                        z_S.append(concat_z_S)
                        
                elif scale == self.scales-1: #last scale
                    right_shape_I = (batch_size, 2**((self.dim-1)*(scale-1))*2**self.dim*init_channels)+tuple([x//2**scale for x in init_res])
                    crop_left_shape_I = right_shape_I
                    z_I.append(self.prior.sample(sample_shape=crop_left_shape_I))

                    right_shape_S = (batch_size, 2**((self.dim-1)*scale)*init_channels)+tuple([x//2**scale for x in init_res])
                    crop_left_shape_S = right_shape_S
                    concat_z_S = self.prior.sample(sample_shape=(scale*crop_left_shape_S[0])+crop_left_shape_S[1:])
                    z_S.append(concat_z_S)
            
            z_short = [z_I, z_S]
            L_pred_short, _ = self.model['condflow'](L=[], z=z_short, D=D, reverse=True, shortcut=True)
            return L_pred_short





        

        
        
        
        
        
        