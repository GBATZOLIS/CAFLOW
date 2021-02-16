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
import torchvision
import numpy as np

class CAFlow(pl.LightningModule):
    def __init__(self, opts):
        super(CAFlow, self).__init__()
        self.save_hyperparameters()
        
        self.train_shortcut = opts.train_shortcut
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
        self.model['rflow'] = UnconditionalFlow(channels=opts.data_channels, dim=opts.data_dim, scales=opts.model_scales, 
                                                scale_depth=opts.model_scale_depth, quants=opts.r_quants)
        self.model['tflow'] = UnconditionalFlow(channels=opts.data_channels, dim=opts.data_dim, scales=opts.model_scales, 
                                                scale_depth=opts.model_scale_depth, quants=opts.t_quants)
        self.model['condflow'] = SharedConditionalFlow(channels=opts.data_channels, dim=opts.data_dim, scales=opts.model_scales, scale_depth=opts.model_scale_depth)
        
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
        
        #optimiser settings
        self.learning_rate = opts.learning_rate

    def forward(self, Y, shortcut=True):
        return self.sample(Y, shortcut=shortcut)

    def logjoint(self, Y, I, shortcut=False, scaled=True):
        D, rlogprior, rlogdet = self.model['rflow'](y=Y)
        L, _, tlogdet = self.model['tflow'](y=I)
        Z_cond, condlogprior, condlogdet = self.model['condflow'](L=L, z=[], D=D, reverse=False, shortcut=shortcut)

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
    
    
    def validation_step(self, batch, batch_idx):
        metric_dict = {}
        
        Y, I = batch
        neg_avg_scaled_logjoint = -1*torch.mean(self.logjoint(Y, I, shortcut=self.train_shortcut, scaled=True))
        loss = neg_avg_scaled_logjoint
        metric_dict['val_loss'] = loss
        
        I_sample =  self.sample(Y)
        metric_dict['val_rec_loss'] = torch.mean(torch.abs(I-I_sample))
        
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
                    sampled_image = self(Y)
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
        
        return metric_dict
            
    def validation_epoch_end(self, outputs):
        print_dict = {}
        for output in outputs:
            for key in output.keys():
                if key not in print_dict:
                    print_dict[key]=[]
                    
                print_dict[key].append(output[key])
        
        print('------- current epoch: %d -------' % self.current_epoch)
        for key in print_dict.keys():
            print_dict[key] = torch.mean(torch.tensor(print_dict[key])).item()
            print('%s : %.12f' % (key, print_dict[key]))
        
        self.log_dict(print_dict)
            
       
    def test_step(self, batch, batch_idx):
        Y, I = batch
        neg_avg_logjoint = -1*torch.mean(self.logjoint(Y, I, shortcut=self.train_shortcut))
        loss = neg_avg_logjoint
        
        I_sample =  self.sample(Y)
        test_rec_loss = torch.mean(torch.abs(I-I_sample))
        
        B = I.shape[0]
        raw_length = 1+self.num_val_samples+1
        all_images = torch.zeros(tuple([B*raw_length,]) + I.shape[1:])
    
        for i in range(B):
            all_images[i*raw_length] = Y[i]
            all_images[(i+1)*raw_length-1] = I[i]
            
        # generate images
        with torch.no_grad():
            for j in range(1, self.num_val_samples+1):
                sampled_image = self(Y)
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
        
        torchvision.utils.save_image(tensor=grid, fp='lightning_logs/version_7/%d.png' % batch_idx)
        
        metric_dict = {'test_loss': loss, 'test_rec_loss': test_rec_loss}
        self.log_dict(metric_dict)
        return metric_dict
        
    def test_epoch_end(self, outputs):
        print_dict = {}
        for output in outputs:
            for key in output.keys():
                if key not in print_dict:
                    print_dict[key]=[]
                    
                print_dict[key].append(output[key])
        
        print('----')
        for key in print_dict.keys():
            print_dict[key] = torch.mean(torch.tensor(print_dict[key])).item()
            print('%s : %.3f' % (key, print_dict[key]))
        
        self.log_dict(print_dict)
        
        
    def configure_optimizers(self,):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]
    
    #@torch.no_grad()
    def sample(self, Y, shortcut=True, xtreme_shortcut=False):
        D, _, _ =self.model['rflow'](y=Y)
        #start_shape = D[0].shape #(batchsize, 2*(dim-1)*self.channels, init_res/2, init_res/2, init_res/2)
        batch_size = D[0].shape[0]
        init_res = D[0].shape[2:]
        init_channels = D[0].shape[1]

        if shortcut and not xtreme_shortcut:
            #D = [D_(n-1), D_(n-2), ..., D_2, D_1, D_0]
            #z_I=z[0] #[z_(n-1)^(n-1), z_(n-2)^(n-2), ..., z_2^2, z_1^1, z_0^0]
            #z_S=z[1] #[z_(n-2)^(n-1),      z_(n-3)^(n-1)||z_(n-3)^(n-2),       z_(n-4)^(n-1)||z_(n-4)^(n-2)||z_(n-4)^(n-3), 
                        # ...,              z_1^(n-1)||z_1^(n-2)|| ... ||z_1^2,       z_0^(n-1)||z_0^(n-2)|| ... ||z_0^1     ]

            z_I = []
            z_S = []
            for scale in range(self.scales):
                if scale < self.scales-1:
                    right_shape = (batch_size, 2**((self.dim-1)*scale)*init_channels)+tuple([x//2**scale for x in init_res])
                    crop_left_shape = (batch_size, 2**(self.dim-1)*right_shape[1])+tuple([x//2 for x in right_shape[2:]])
                    if scale == 0:
                        z_I.append(self.prior.sample(sample_shape=crop_left_shape).to(self.device))
                    else:
                        z_I.append(self.prior.sample(sample_shape=crop_left_shape).to(self.device))
                        concat_z_S = self.prior.sample(sample_shape=tuple([scale*crop_left_shape[0],])+crop_left_shape[1:]).to(self.device)
                        z_S.append(concat_z_S)
                        
                elif scale == self.scales-1: #last scale
                    right_shape_I = (batch_size, 2**((self.dim-1)*(scale-1))*2**self.dim*init_channels)+tuple([x//2**scale for x in init_res])
                    crop_left_shape_I = (batch_size, 2**self.dim*right_shape_I[1])+tuple([x//2 for x in right_shape_I[2:]])
                    z_I.append(self.prior.sample(sample_shape=crop_left_shape_I).to(self.device))

                    right_shape_S = (batch_size, 2**((self.dim-1)*scale)*init_channels)+tuple([x//2**scale for x in init_res])
                    crop_left_shape_S = (batch_size, 2**self.dim*right_shape_S[1])+tuple([x//2 for x in right_shape_S[2:]])
                    concat_z_S = self.prior.sample(sample_shape=tuple([scale*crop_left_shape_S[0],])+crop_left_shape_S[1:]).to(self.device)
                    z_S.append(concat_z_S)
            
            z_short = [z_I, z_S]
            L_pred_short, _ = self.model['condflow'](L=[], z=z_short, D=D, reverse=True, shortcut=True)
            I_sample, logdet = self.model['tflow'](z=L_pred_short, reverse=True)
            
            return I_sample
        
        elif shortcut and xtreme_shortcut:
            z_I = []
            z_S = []
            for scale in range(self.scales):
                if scale < self.scales-1:
                    right_shape = (batch_size, 2**((self.dim-1)*scale)*init_channels)+tuple([x//2**scale for x in init_res])
                    crop_left_shape = (batch_size, 2**(self.dim-1)*right_shape[1])+tuple([x//2 for x in right_shape[2:]])
                    if scale == 0:
                        z_I.append(self.prior.sample(sample_shape=crop_left_shape).to(self.device))
                    else:
                        z_I.append(self.prior.sample(sample_shape=crop_left_shape).to(self.device))
                        concat_z_S = self.prior.sample(sample_shape=crop_left_shape).to(self.device)
                        z_S.append(concat_z_S)
                        
                elif scale == self.scales-1: #last scale
                    right_shape_I = (batch_size, 2**((self.dim-1)*(scale-1))*2**self.dim*init_channels)+tuple([x//2**scale for x in init_res])
                    crop_left_shape_I = (batch_size, 2**self.dim*right_shape_I[1])+tuple([x//2 for x in right_shape_I[2:]])
                    z_I.append(self.prior.sample(sample_shape=crop_left_shape_I).to(self.device))

                    right_shape_S = (batch_size, 2**((self.dim-1)*scale)*init_channels)+tuple([x//2**scale for x in init_res])
                    crop_left_shape_S = (batch_size, 2**self.dim*right_shape_S[1])+tuple([x//2 for x in right_shape_S[2:]])
                    concat_z_S = self.prior.sample(sample_shape=crop_left_shape_S).to(self.device)
                    z_S.append(concat_z_S)
            
            z_xtreme_short = [z_I, z_S]
            L_pred_short, _ = self.model['condflow'](L=[], z=z_xtreme_short, D=D, reverse=True, shortcut=True, xtreme_shortcut=True)
            I_sample, logdet = self.model['tflow'](z=L_pred_short, reverse=True)
            
            return I_sample





        

        
        
        
        
        
        