#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 17:29:09 2021

@author: gbatz97
"""

from caflow.models.modules.mri_to_pet.UnconditionalFlow import UnconditionalFlow
from caflow.models.modules.mri_to_pet.SharedConditionalFlow import SharedConditionalFlow
from caflow.models.modules.mri_to_pet.UnsharedConditionalFlow import UnsharedConditionalFlow
from caflow.models.UFlow import UFlow
import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision
import numpy as np

class CAFlow(pl.LightningModule):
    def __init__(self, opts):
        super(CAFlow, self).__init__()
        self.save_hyperparameters()

        self.shared = opts.shared
        if not opts.shared:
            self.train_shortcut = False
            self.val_shortcut = False
        else:
            self.train_shortcut = opts.train_shortcut
            self.val_shortcut = opts.val_shortcut

        self.dim = opts.data_dim
        self.scales = opts.model_scales
        self.channels = opts.data_channels
        
        #validation conditional sampling settings
        self.num_val_samples = opts.num_val_samples #num of validation samples
        self.sample_padding = opts.sample_padding #bool
        self.sample_normalize = opts.sample_normalize #bool
        self.sample_norm_range = opts.sample_norm_range #tuple range
        self.sample_scale_each = opts.sample_scale_each #bool
        self.sample_pad_value = opts.sample_pad_value #pad value

        self.model = nn.ModuleDict()

        if opts.pretrain == 'conditional':
            assert opts.rflow_checkpoint is not None, 'opts.rflow_checkpoint is not set.'
            assert opts.tflow_checkpoint is not None, 'opts.tflow_checkpoint is not set.'
            assert opts.cflow_checkpoint is None, 'Conditional flow is pretrained. opts.cflow_checkpoint is set.'

        if opts.rflow_checkpoint is not None:
            self.model['rflow'] = UFlow.load_from_checkpoint(opts.rflow_checkpoint)
        else:
            self.model['rflow'] = UnconditionalFlow(channels=opts.data_channels, dim=opts.data_dim, 
                                                    scales=opts.model_scales, scale_depth=opts.rflow_scale_depth, 
                                                    quants=opts.r_quants, vardeq_depth=opts.vardeq_depth)
        
        if opts.tflow_checkpoint is not None:
            self.model['tflow'] = UFlow.load_from_checkpoint(opts.tflow_checkpoint)
        else:
            self.model['tflow'] = UnconditionalFlow(channels=opts.data_channels, dim=opts.data_dim, 
                                                    scales=opts.model_scales, scale_depth=opts.rflow_scale_depth, 
                                                    quants=opts.r_quants, vardeq_depth=opts.vardeq_depth)

        if opts.shared:
            self.model['SharedConditionalFlow'] = SharedConditionalFlow(channels=opts.data_channels, \
                                                                        dim=opts.data_dim, scales=opts.model_scales, \
                                                                        shared_scale_depth=opts.s_cond_s_scale_depth, \
                                                                        unshared_scale_depth=opts.s_cond_u_scale_depth)
        else:
            self.model['UnsharedConditionalFlow'] = UnsharedConditionalFlow(channels=opts.data_channels, \
                                                                            dim=opts.data_dim, \
                                                                            scales=opts.model_scales, \
                                                                            scale_depth=opts.u_cond_scale_depth)
        if opts.pretrain == 'conditional':
            self.model['rflow'].freeze()
            self.model['tflow'].freeze()      

        #set the prior distribution for the latents
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
        
        #optimiser settings
        self.learning_rate = opts.learning_rate

    def forward(self, Y, shortcut=True):
        return self.sample(Y, shortcut=shortcut)

    def logjoint(self, Y, I, shortcut=False, scaled=True):
        D, rlogprior, rlogdet = self.model['rflow'](y=Y)
        L, _, tlogdet = self.model['tflow'](y=I)

        if self.shared:
            Z_cond, condlogprior, condlogdet = self.model['SharedConditionalFlow'](L=L, z=[], D=D, reverse=False, shortcut=shortcut)
        else:
            Z_cond, condlogprior, condlogdet = self.model['UnsharedConditionalFlow'](L=L, z=[], D=D, reverse=False)

        logjoint = rlogprior + rlogdet + tlogdet + condlogprior + condlogdet

        if scaled:
            scaled_logjoint = logjoint*np.log2(np.exp(1))/(np.prod(Y.shape[1:])*np.prod(I.shape[1:]))
            return scaled_logjoint
        else:
            return logjoint
    
    def training_step(self, batch, batch_idx):
        Y, I = batch
        neg_avg_scaled_logjoint = -1*torch.mean(self.logjoint(Y, I, shortcut=self.train_shortcut, scaled=True))
        loss = neg_avg_scaled_logjoint
        
        #logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def training_step_end(self, training_step_outputs):
        return training_step_outputs.sum()
    
    def validation_step(self, batch, batch_idx):
        Y, I = batch
        neg_avg_scaled_logjoint = -1*torch.mean(self.logjoint(Y, I, shortcut=self.val_shortcut, scaled=True))
        val_loss = neg_avg_scaled_logjoint
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, sync_dist=True)
        
        I_sample =  self.sample(Y, shortcut=self.val_shortcut)
        val_rec_loss = torch.mean(torch.abs(I-I_sample))
        self.log('val_rec_loss', val_rec_loss, on_step=True, on_epoch=True, sync_dist=True)
        
        B = I.shape[0]
        if batch_idx==0:
            raw_length = 1+self.num_val_samples+1
            all_images = torch.zeros(tuple([B*raw_length,]) + I.shape[1:])
            
            for i in range(B):
                all_images[i*raw_length] = Y[i]
                all_images[(i+1)*raw_length-1] = I[i]
                
            # generate images
            with torch.no_grad():
                for j in range(1, self.num_val_samples+1):
                    sampled_image = self.sample(Y, shortcut=self.val_shortcut)
                    for i in range(B):
                        all_images[i*raw_length+j]=sampled_image[i]
               
            grid = torchvision.utils.make_grid(
                tensor=all_images,
                nrow = raw_length, #Number of images displayed in each row of the grid
                padding=self.sample_padding,
                normalize=self.sample_normalize,
                range=self.sample_norm_range,
                scale_each=self.sample_scale_each,
                pad_value=self.sample_pad_value,
            )
            str_title = 'val_samples_epoch_%d__batch_%d' % (self.current_epoch, batch_idx)
            self.logger.experiment.add_image(str_title, grid, self.current_epoch)

    def configure_optimizers(self,):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.999)
        return [optimizer], [scheduler]
    
    #@torch.no_grad()
    def sample(self, Y, shortcut=True, xtreme_shortcut=False):
        D, _, _ =self.model['rflow'](y=Y) #D = [D_(n-1), D_(n-2), ..., D_2, D_1, D_0]

        #Use shape of D[0] to calculate dynamically the shapes of sampled tensors of the conditional flows. # this should be changed for super-resolution.
        #base_shape = D[0].shape #(batchsize, 2*(dim-1)*self.channels, init_res/2, init_res/2, init_res/2)
        batch_size = D[0].shape[0]
        init_res = D[0].shape[2:]
        init_channels = D[0].shape[1]

        if not shortcut and not xtreme_shortcut:
            z_normal = self.generate_z_cond(D[0], shortcut=False)
            if self.shared:
                L_pred, _ = self.model['SharedConditionalFlow'](L=[], z=z_normal, D=D, reverse=True, shortcut=False)
            else:
                L_pred, _ = self.model['UnsharedConditionalFlow'](L=[], z=z_normal, D=D, reverse=True)
            I_sample, _ = self.model['tflow'](z=L_pred, reverse=True)
            return I_sample

        elif shortcut and not xtreme_shortcut:
            z_short = self.generate_z_cond(D[0], shortcut=True)
            L_pred_short, _ = self.model['SharedConditionalFlow'](L=[], z=z_short, D=D, reverse=True, shortcut=True)
            I_sample, logdet = self.model['tflow'](z=L_pred_short, reverse=True)
            return I_sample
        
        elif shortcut and xtreme_shortcut:
            z_xtreme_short = self.generate_z_cond(D[0], shortcut=True, xtreme_shortcut=True)
            L_pred_short, _ = self.model['SharedConditionalFlow'](L=[], z=z_xtreme_short, D=D, reverse=True, shortcut=True, xtreme_shortcut=True)
            I_sample, logdet = self.model['tflow'](z=L_pred_short, reverse=True)
            
            return I_sample

    def generate_z_cond(self, D0, shortcut, xtreme_shortcut=False):
        """Generates the sampled tensors for the conditional flows"""
        # D = [D_(n-1), D_(n-2), ..., D_2, D_1, D_0]
        # The sampled tensors for the shared conditional flows are the following:
        # z_I=z[0] - [z_(n-1)^(n-1), z_(n-2)^(n-2), ..., z_2^2, z_1^1, z_0^0]
        # z_S=z[1] - [z_(n-2)^(n-1),      z_(n-3)^(n-1)||z_(n-3)^(n-2),       z_(n-4)^(n-1)||z_(n-4)^(n-2)||z_(n-4)^(n-3), 
        #             ...,              z_1^(n-1)||z_1^(n-2)|| ... ||z_1^2,       z_0^(n-1)||z_0^(n-2)|| ... ||z_0^1     ]
        # This function creates the sampled tensors for the shortcut option. 
        # We provide the option to convert the shortcut version to the normal version using shortcut=False.

        #Use shape of D[0] to calculate dynamically the shapes of sampled tensors of the conditional flows.
        #base_shape = D[0].shape #(batchsize, 2*(dim-1)*self.channels, init_res/2, init_res/2, init_res/2)
        batch_size = D0.shape[0] #D0 is D_(n-1): the first element of the list of tensors D.
        init_res = D0.shape[2:]
        init_channels = D0.shape[1]

        z_I = []
        z_S = []
        if xtreme_shortcut:
            for scale in range(self.scales):
                if scale < self.scales-1:
                    right_shape = (batch_size, 2**((self.dim-1)*scale)*init_channels)+tuple([x//2**scale for x in init_res])
                    crop_left_shape = (batch_size, 2**(self.dim-1)*right_shape[1])+tuple([x//2 for x in right_shape[2:]])
                    if scale == 0:
                        z_I.append(self.prior.sample(sample_shape=crop_left_shape).type_as(D0))
                    else:
                        z_I.append(self.prior.sample(sample_shape=crop_left_shape).type_as(D0))
                        concat_z_S = self.prior.sample(sample_shape=crop_left_shape).type_as(D0)
                        z_S.append(concat_z_S)
                        
                elif scale == self.scales-1: #last scale
                    right_shape_I = (batch_size, 2**((self.dim-1)*(scale-1))*2**self.dim*init_channels)+tuple([x//2**scale for x in init_res])
                    crop_left_shape_I = (batch_size, 2**self.dim*right_shape_I[1])+tuple([x//2 for x in right_shape_I[2:]])
                    z_I.append(self.prior.sample(sample_shape=crop_left_shape_I).type_as(D0))

                    right_shape_S = (batch_size, 2**((self.dim-1)*scale)*init_channels)+tuple([x//2**scale for x in init_res])
                    crop_left_shape_S = (batch_size, 2**self.dim*right_shape_S[1])+tuple([x//2 for x in right_shape_S[2:]])
                    concat_z_S = self.prior.sample(sample_shape=crop_left_shape_S).type_as(D0)
                    z_S.append(concat_z_S)
            
            z_xtreme_short = [z_I, z_S]
            return z_xtreme_short
        else:
            for scale in range(self.scales):
                if scale < self.scales-1:
                    right_shape = (batch_size, 2**((self.dim-1)*scale)*init_channels)+tuple([x//2**scale for x in init_res])
                    crop_left_shape = (batch_size, 2**(self.dim-1)*right_shape[1])+tuple([x//2 for x in right_shape[2:]])
                    if scale == 0:
                        z_I.append(self.prior.sample(sample_shape=crop_left_shape).type_as(D0))
                    else:
                        z_I.append(self.prior.sample(sample_shape=crop_left_shape).type_as(D0))
                        concat_z_S = self.prior.sample(sample_shape=tuple([scale*crop_left_shape[0],])+crop_left_shape[1:]).type_as(D0)
                        z_S.append(concat_z_S)
                        
                elif scale == self.scales-1: #last scale
                    right_shape_I = (batch_size, 2**((self.dim-1)*(scale-1))*2**self.dim*init_channels)+tuple([x//2**scale for x in init_res])
                    crop_left_shape_I = (batch_size, 2**self.dim*right_shape_I[1])+tuple([x//2 for x in right_shape_I[2:]])
                    z_I.append(self.prior.sample(sample_shape=crop_left_shape_I).type_as(D0))

                    right_shape_S = (batch_size, 2**((self.dim-1)*scale)*init_channels)+tuple([x//2**scale for x in init_res])
                    crop_left_shape_S = (batch_size, 2**self.dim*right_shape_S[1])+tuple([x//2 for x in right_shape_S[2:]])
                    concat_z_S = self.prior.sample(sample_shape=tuple([scale*crop_left_shape_S[0],])+crop_left_shape_S[1:]).type_as(D0)
                    z_S.append(concat_z_S)
            
            z_short = [z_I, z_S]
        
            if shortcut:
                return z_short
            else:
                return self.convert_shortcut_to_normal(z_short)

    def convert_shortcut_to_normal(self, z_short):
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

    def convert_normal_to_shortcut(self, z_normal):
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




        

        
        
        
        
        
        