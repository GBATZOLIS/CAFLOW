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
from caflow.utils.processing import normalise
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
        self.channels = opts.data_channels
        
        if len(opts.resolution) > 1 :
            self.resolution = opts.resolution
        else: 
            self.resolution = [opts.resolution[0] for _ in range(self.dim)]

        self.scales = opts.model_scales
        
        #validation conditional sampling settings
        self.num_val_samples = opts.num_val_samples #num of validation samples
        self.sample_padding = opts.sample_padding #bool
        self.sample_normalize = opts.sample_normalize #bool
        self.sample_norm_range = opts.sample_norm_range #tuple range
        self.sample_scale_each = opts.sample_scale_each #bool
        self.sample_pad_value = opts.sample_pad_value #pad value
        self.sampling_temperatures = opts.sampling_temperatures #list of sampling temperatures

        self.model = nn.ModuleDict()

        if opts.pretrain == 'conditional':
            assert opts.rflow_checkpoint is not None, 'opts.rflow_checkpoint is not set.'
            assert opts.tflow_checkpoint is not None, 'opts.tflow_checkpoint is not set.'
            assert opts.cflow_checkpoint is None, 'Conditional flow is pretrained. opts.cflow_checkpoint is set.'

        if opts.rflow_checkpoint is not None:
            self.model['rflow'] = UFlow.load_from_checkpoint(opts.rflow_checkpoint)
        else:
            self.model['rflow'] = UFlow(dim=opts.data_dim, scales=opts.model_scales, channels=opts.data_channels, 
                                        resolution=opts.resolution, scale_depth=opts.rflow_scale_depth,
                                        use_inv_scaling=opts.use_inv_scaling, scaling_range=opts.r_domain_range, 
                                        use_dequantisation=opts.use_dequantisation, quants=opts.r_quants, 
                                        vardeq_depth=opts.vardeq_depth, opts=opts)
        
        if opts.tflow_checkpoint is not None:
            self.model['tflow'] = UFlow.load_from_checkpoint(opts.tflow_checkpoint)
        else:
            self.model['tflow'] = UFlow(dim=opts.data_dim, scales=opts.model_scales, channels=opts.data_channels, 
                                        resolution=opts.resolution, scale_depth=opts.tflow_scale_depth,
                                        use_inv_scaling=opts.use_inv_scaling, scaling_range=opts.t_domain_range, 
                                        use_dequantisation=opts.use_dequantisation, quants=opts.t_quants, 
                                        vardeq_depth=opts.vardeq_depth, opts=opts)

        if opts.shared:
            self.model['SharedConditionalFlow'] = SharedConditionalFlow(channels=opts.data_channels, 
                                                                        dim=opts.data_dim,
                                                                        resolution = self.resolution,
                                                                        scales=opts.model_scales, 
                                                                        shared_scale_depth=opts.s_cond_s_scale_depth, 
                                                                        unshared_scale_depth=opts.s_cond_u_scale_depth,
                                                                        nn_settings=self.create_nn_settings(opts))
        else:
            self.model['UnsharedConditionalFlow'] = UnsharedConditionalFlow(channels=opts.data_channels, 
                                                                            dim=opts.data_dim, 
                                                                            resolution = self.resolution,
                                                                            scales=opts.model_scales, 
                                                                            scale_depth=opts.u_cond_scale_depth,
                                                                            nn_settings=self.create_nn_settings(opts))
        if opts.pretrain == 'conditional':
            self.model['rflow'].freeze()
            self.model['tflow'].freeze()      

        #set the prior distribution for the latents
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

        #loss function settings
        self.lamda = opts.lamda
        self.lambda_rec = opts.lambda_rec

        #optimiser settings
        self.learning_rate = opts.learning_rate
        self.level_off_factor = opts.level_off_factor
        self.level_off_steps = opts.level_off_steps
        self.use_warm_up = opts.use_warm_up
        self.warm_up = opts.warm_up
        self.gamma = opts.gamma
        self.use_ema = opts.use_ema
        
    def create_nn_settings(self, opts):
        nn_settings={'nn_type':opts.nn_type, 'c_hidden_factor':opts.CAFLOW_c_hidden_factor, \
            'drop_prob':opts.drop_prob, 'num_blocks':opts.num_blocks, 'use_attn':opts.use_attn,\
            'num_components':opts.num_components, 'num_channels_factor':opts.num_channels_factor, 'dim':opts.data_dim}
        return nn_settings

    def forward(self, Y, shortcut=True):
        return self.sample(Y, shortcut=shortcut)

    def logjoint(self, Y, I, shortcut=False, scaled=True):
        D, rlogprior, rlogdet = self.model['rflow'](y=Y)
        L, _, tlogdet = self.model['tflow'](y=I)

        if self.shared:
            Z_cond, condlogprior, condlogdet = self.model['SharedConditionalFlow'](L=L, z=[], D=D, reverse=False, shortcut=shortcut)
        else:
            Z_cond, condlogprior, condlogdet = self.model['UnsharedConditionalFlow'](L=L, z=[], D=D, reverse=False)

        logjoint = self.lamda*(rlogprior + rlogdet) + tlogdet + condlogprior + condlogdet

        if scaled:
            #scaled_logjoint = logjoint*np.log2(np.exp(1))/(np.prod(Y.shape[1:])*np.prod(I.shape[1:]))
            scaled_logjoint = logjoint/(np.prod(Y.shape[1:])*np.prod(I.shape[1:]))
            return scaled_logjoint, [D, Z_cond, L]
        else:
            return logjoint, [D, Z_cond, L]
    
    def training_step(self, batch, batch_idx):
        Y, I = batch
        scaled_logjoint, encodings = self.logjoint(Y, I, shortcut=self.train_shortcut, scaled=True)
        neg_avg_scaled_logjoint = -1*torch.mean(scaled_logjoint)

        if self.lambda_rec is not None:
            D, z_cond = encodings[0], encodings[1]
            z_normal = self.generate_z_cond(D[0], shortcut=False)
            z_normal[-1][0] = z_cond[-1][0].clone()
            del z_cond

            L_pred, _ = self.model['UnsharedConditionalFlow'](L=[], z=z_normal, D=D, reverse=True)
            I_rec, _ = self.model['tflow'](z=L_pred, reverse=True)
            train_rec_loss = torch.mean(torch.abs(I-I_rec))
            self.log('train_rec_loss', train_rec_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            loss = neg_avg_scaled_logjoint + self.lambda_rec * train_rec_loss
        else:
            #code for checking invertibility of the analytically invertible functions of the framework.
            '''
            def mean_abs(X:list, Y:list):
                result = 0.
                for x, y in zip(X,Y):
                    result += torch.mean(torch.abs(x - y))
                return result

            with torch.no_grad():
                D, Z_cond, L = encodings[0], encodings[1], encodings[2]
                Y_dash, _ = self.model['rflow'](z=D, reverse=True)
                I_dash, _ = self.model['tflow'](z=L, reverse=True)
                if self.shared:
                    L_dash, _ = self.model['SharedConditionalFlow'](L=[], z=Z_cond, D=D, reverse=True, shortcut=self.train_shortcut)
                else:
                    L_dash, _ = self.model['UnsharedConditionalFlow'](L=[], z=Z_cond, D=D, reverse=True)
                
                #print(torch.mean(torch.abs(Y-Y_dash)))
                self.log('r_flow_rec', torch.mean(torch.abs(Y-Y_dash)), on_step=True, on_epoch=True, prog_bar=True, logger=True)
                self.log('t_flow_rec', torch.mean(torch.abs(I-I_dash)), on_step=True, on_epoch=True, prog_bar=True, logger=True)
                self.log('cond_flow_rec', mean_abs(L, L_dash), on_step=True, on_epoch=True, prog_bar=True, logger=True)
            '''
            loss = neg_avg_scaled_logjoint
        
        #logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def training_step_end(self, training_step_outputs):
        return training_step_outputs.sum()
    
    def validation_step(self, batch, batch_idx):
        Y, I = batch
        neg_avg_scaled_logjoint = -1*torch.mean(self.logjoint(Y, I, shortcut=self.val_shortcut, scaled=True)[0])
        val_loss = neg_avg_scaled_logjoint
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, sync_dist=True)
        
        I_sample =  self.sample(Y, shortcut=self.val_shortcut)
        val_rec_loss = torch.mean(torch.abs(I-I_sample))
        self.log('val_rec_loss', val_rec_loss, on_step=True, on_epoch=True, sync_dist=True)

        B = Y.shape[0]
        
        if self.dim == 2 and I.size(1) == 3:
            raw_length = 1+self.num_val_samples+1
            all_images = torch.zeros(tuple([B*raw_length,]) + I.shape[1:])
                
            for i in range(B):
                all_images[i*raw_length] = Y[i]
                all_images[(i+1)*raw_length-1] = I[i]
                
            for sampling_T in self.sampling_temperatures:
                # generate images
                with torch.no_grad():
                    for j in range(1, self.num_val_samples+1):
                        sampled_image = self.sample(Y, shortcut=self.val_shortcut, T=sampling_T)
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
                str_title = 'val_samples_epoch_%d_T_%.2f' % (self.current_epoch, sampling_T)
                self.logger.experiment.add_image(str_title, grid, self.current_epoch)
            
        elif self.dim == 2 and I.size(1) != 3: #sliced medical scans -> we treat the third dimension as the channel dimension (number of channels more than 3)
            #Y, I shape: (batchsize, channels, x1, x2)
            raw_length = 1+self.num_val_samples+1
            all_images = torch.zeros(tuple([B*raw_length, 1, I.shape[2], I.shape[3]]))
            for i in range(B):
                all_images[i*raw_length] = normalise(Y[i, self.channels//2, :, :]).unsqueeze(0)
                all_images[(i+1)*raw_length-1] = normalise(I[i, self.channels//2, :, :]).unsqueeze(0)
                
            for sampling_T in self.sampling_temperatures:
                # generate images
                for j in range(1, self.num_val_samples+1):
                    sampled_image = self.sample(Y, shortcut=self.val_shortcut, T=sampling_T)
                    for i in range(B):
                        all_images[i*raw_length+j] = normalise(sampled_image[i, self.channels//2, :, :]).unsqueeze(0)
                    
                grid = torchvision.utils.make_grid(tensor = all_images, nrow = raw_length, padding=self.sample_padding, normalize=False, pad_value=self.sample_pad_value)
                str_title = 'val_samples_epoch_%d_T_%.2f_middlecut_dim3' % (self.current_epoch, sampling_T)
                self.logger.experiment.add_image(str_title, grid, self.current_epoch)

        elif self.dim == 3:
            #Y, I shape: (batchsize, 1, x1, x2, x3) - (0, 1, 2, 3, 4)
            #We are going to display slices of the 3D reconstructed PET image. 
            #We will additionally save the synthetic and real images for further evaluation outside tensorboard.
                
            def generate_paired_video(Y, I, num_samples, dim, epoch, batch):
                #dim: the sliced dimension (choices: 1,2,3)
                B = Y.size(0)
                raw_length = 1+num_samples+1
                frames = Y.size(dim+1)
                video_grid = []
                for frame in range(frames):
                    if dim==1:
                        dim_cut = torch.zeros(tuple([B*raw_length, 1, I.shape[3], I.shape[4]]))
                    elif dim==2:
                        dim_cut = torch.zeros(tuple([B*raw_length, 1, I.shape[2], I.shape[4]]))
                    elif dim==3:
                        dim_cut = torch.zeros(tuple([B*raw_length, 1, I.shape[2], I.shape[3]]))

                    for i in range(B):
                        if dim==1:
                            dim_cut[i*raw_length] = normalise(Y[i, 0, frame, :, :]).unsqueeze(0)
                            dim_cut[(i+1)*raw_length-1] = normalise(I[i, 0, frame, :, :]).unsqueeze(0)
                        elif dim==2:
                            dim_cut[i*raw_length] = normalise(Y[i, 0, :, frame, :]).unsqueeze(0)
                            dim_cut[(i+1)*raw_length-1] = normalise(I[i, 0, :, frame, :]).unsqueeze(0)
                        elif dim==3:
                            dim_cut[i*raw_length] = normalise(Y[i, 0, :, :, frame]).unsqueeze(0)
                            dim_cut[(i+1)*raw_length-1] = normalise(I[i, 0, :, :, frame]).unsqueeze(0)

                    grid_cut = torchvision.utils.make_grid(tensor=dim_cut, nrow=raw_length, 
                                                    padding=self.sample_padding, normalize=False, pad_value=self.sample_pad_value)
                    #print(grid_cut.size())
                    video_grid.append(grid_cut)

                video_grid = torch.stack(video_grid, dim=0).unsqueeze(0)
                print(video_grid.size())

                str_title = 'paired_video_epoch_%d_batch_%d_dim_%d' % (epoch, batch, dim)
                self.logger.experiment.add_video(str_title, video_grid, self.current_epoch)

            '''
            generate_paired_video(Y, I, 0, 1, self.current_epoch, batch_idx)
            generate_paired_video(Y, I, 0, 2, self.current_epoch, batch_idx)
            generate_paired_video(Y, I, 0, 3, self.current_epoch, batch_idx)
            '''

            
            raw_length = 1 + self.num_val_samples + 1
            dim1cut = torch.zeros(tuple([B*raw_length, 1, I.shape[3], I.shape[4]]))
            dim2cut = torch.zeros(tuple([B*raw_length, 1, I.shape[2], I.shape[4]]))
            dim3cut = torch.zeros(tuple([B*raw_length, 1, I.shape[2], I.shape[3]]))

            for i in range(B):
                dim1cut[i*raw_length] = normalise(Y[i, 0, Y.shape[2]//2, :, :]).unsqueeze(0)
                dim1cut[(i+1)*raw_length-1] = normalise(I[i, 0, I.shape[2]//2, :, :]).unsqueeze(0)
                dim2cut[i*raw_length] = normalise(Y[i, 0, :, Y.shape[3]//2, :]).unsqueeze(0)
                dim2cut[(i+1)*raw_length-1] = normalise(I[i, 0, :, I.shape[3]//2, :]).unsqueeze(0)
                dim3cut[i*raw_length] = normalise(Y[i, 0, :, :, Y.shape[4]//2]).unsqueeze(0)
                dim3cut[(i+1)*raw_length-1] = normalise(I[i, 0, :, :, I.shape[4]//2]).unsqueeze(0)

            for sampling_T in self.sampling_temperatures:
                # generate images
                with torch.no_grad():
                    for j in range(1, self.num_val_samples+1):
                        sampled_image = self.sample(Y, shortcut=self.val_shortcut, T=sampling_T)
                        for i in range(B):
                            dim1cut[i*raw_length+j] = normalise(sampled_image[i, 0, I.shape[2]//2, :, :]).unsqueeze(0)
                            dim2cut[i*raw_length+j] = normalise(sampled_image[i, 0, :, I.shape[3]//2, :]).unsqueeze(0)
                            dim3cut[i*raw_length+j] = normalise(sampled_image[i, 0, :, :, I.shape[4]//2]).unsqueeze(0)
                    
                #--------------------------------------------------------------------------------------------
                grid = torchvision.utils.make_grid(tensor=dim1cut, nrow = raw_length, padding=self.sample_padding, normalize=False, pad_value=self.sample_pad_value)
                str_title = 'val_samples_epoch_%d_T_%.2f_cut_dim1' % (self.current_epoch, sampling_T)
                self.logger.experiment.add_image(str_title, grid, self.current_epoch)

                #--------------------------------------------------------------------------------------------
                grid = torchvision.utils.make_grid(tensor=dim2cut, nrow = raw_length, padding=self.sample_padding, normalize=False, pad_value=self.sample_pad_value)
                str_title = 'val_samples_epoch_%d_T_%.2f_cut_dim2' % (self.current_epoch, sampling_T)
                self.logger.experiment.add_image(str_title, grid, self.current_epoch)

                #--------------------------------------------------------------------------------------------
                grid = torchvision.utils.make_grid(tensor=dim3cut, nrow = raw_length, padding=self.sample_padding, normalize=False, pad_value=self.sample_pad_value)
                str_title = 'val_samples_epoch_%d_T_%.2f_cut_dim3' % (self.current_epoch, sampling_T)
                self.logger.experiment.add_image(str_title, grid, self.current_epoch)
                #--------------------------------------------------------------------------------------------
            

    def configure_optimizers(self,):
        class scheduler_lambda_function:
            def __init__(self, use_warm_up, warm_up, level_off_steps, level_off_factor, gamma, current_epoch):
                self.use_warm_up = use_warm_up
                self.warm_up = warm_up
                self.level_off_steps = level_off_steps
                self.level_off_factor = level_off_factor
                self.gamma = gamma
                self.current_epoch = current_epoch
            
            def calculate_exponent_factor(self, s, level_off_steps):
                if s < level_off_steps[0]:
                    return 0
                elif s >= level_off_steps[-1]:
                    return len(level_off_steps)
                else:
                    exponent=None
                    for i in range(len(level_off_steps)-1):
                        if s >= level_off_steps[i] and s < level_off_steps[i+1]:
                            exponent = i+1
                    return exponent

            def __call__(self, s):
                if self.use_warm_up:
                    if s < self.warm_up:
                        return s / self.warm_up
                    else:
                        exponent = self.calculate_exponent_factor(s, self.level_off_steps)
                        return self.level_off_factor**exponent*self.gamma**(self.current_epoch)
                else:
                    exponent = self.calculate_exponent_factor(s, self.level_off_steps)
                    return self.level_off_factor**exponent*self.gamma**(self.current_epoch)

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {'scheduler': optim.lr_scheduler.LambdaLR(optimizer, \
                    scheduler_lambda_function(self.use_warm_up, self.warm_up, self.level_off_steps, self.level_off_factor, self.gamma, self.current_epoch)),
                    'interval': 'step'}  # called after each training step
                    
        return [optimizer], [scheduler]
    
    #@torch.no_grad()
    def sample(self, Y, shortcut=True, xtreme_shortcut=False, T=1, temperature_list=None):
        D, _, _ =self.model['rflow'](y=Y) #D = [D_(n-1), D_(n-2), ..., D_2, D_1, D_0]

        #Use shape of D[0] to calculate dynamically the shapes of sampled tensors of the conditional flows. # this should be changed for super-resolution.
        #base_shape = D[0].shape #(batchsize, 2*(dim-1)*self.channels, init_res/2, init_res/2, init_res/2)
        #batch_size = D[0].shape[0]
        #init_res = D[0].shape[2:]
        #init_channels = D[0].shape[1]

        if not shortcut and not xtreme_shortcut:
            z_normal = self.generate_z_cond(D[0], shortcut=False, T=T, temperature_list=temperature_list)
            if self.shared:
                L_pred, _ = self.model['SharedConditionalFlow'](L=[], z=z_normal, D=D, reverse=True, shortcut=False)
            else:
                L_pred, _ = self.model['UnsharedConditionalFlow'](L=[], z=z_normal, D=D, reverse=True)
            I_sample, _ = self.model['tflow'](z=L_pred, reverse=True)
            return I_sample

        elif shortcut and not xtreme_shortcut:
            z_short = self.generate_z_cond(D[0], shortcut=True, T=T, temperature_list=temperature_list)
            L_pred_short, _ = self.model['SharedConditionalFlow'](L=[], z=z_short, D=D, reverse=True, shortcut=True)
            I_sample, logdet = self.model['tflow'](z=L_pred_short, reverse=True)
            return I_sample
        
        elif shortcut and xtreme_shortcut:
            z_xtreme_short = self.generate_z_cond(D[0], shortcut=True, xtreme_shortcut=True, T=T)
            L_pred_short, _ = self.model['SharedConditionalFlow'](L=[], z=z_xtreme_short, D=D, reverse=True, shortcut=True, xtreme_shortcut=True)
            I_sample, logdet = self.model['tflow'](z=L_pred_short, reverse=True)
            
            return I_sample

    def generate_z_cond(self, D0, shortcut, xtreme_shortcut=False, T=1, temperature_list=None):
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
                        z_I.append(T*self.prior.sample(sample_shape=crop_left_shape).type_as(D0))
                    else:
                        z_I.append(T*self.prior.sample(sample_shape=crop_left_shape).type_as(D0))
                        concat_z_S = T*self.prior.sample(sample_shape=crop_left_shape).type_as(D0)
                        z_S.append(concat_z_S)
                        
                elif scale == self.scales-1: #last scale
                    right_shape_I = (batch_size, 2**((self.dim-1)*(scale-1))*2**self.dim*init_channels)+tuple([x//2**scale for x in init_res])
                    crop_left_shape_I = (batch_size, 2**self.dim*right_shape_I[1])+tuple([x//2 for x in right_shape_I[2:]])
                    z_I.append(T*self.prior.sample(sample_shape=crop_left_shape_I).type_as(D0))

                    right_shape_S = (batch_size, 2**((self.dim-1)*scale)*init_channels)+tuple([x//2**scale for x in init_res])
                    crop_left_shape_S = (batch_size, 2**self.dim*right_shape_S[1])+tuple([x//2 for x in right_shape_S[2:]])
                    concat_z_S = T*self.prior.sample(sample_shape=crop_left_shape_S).type_as(D0)
                    z_S.append(concat_z_S)
            
            z_xtreme_short = [z_I, z_S]
            return z_xtreme_short
        else:
            for scale in range(self.scales):
                if scale < self.scales-1:
                    right_shape = (batch_size, 2**((self.dim-1)*scale)*init_channels)+tuple([x//2**scale for x in init_res])
                    crop_left_shape = (batch_size, 2**(self.dim-1)*right_shape[1])+tuple([x//2 for x in right_shape[2:]])
                    if scale == 0:
                        z_I.append(T*self.prior.sample(sample_shape=crop_left_shape).type_as(D0))
                    else:
                        z_I.append(T*self.prior.sample(sample_shape=crop_left_shape).type_as(D0))
                        concat_z_S = T*self.prior.sample(sample_shape=tuple([scale*crop_left_shape[0],])+crop_left_shape[1:]).type_as(D0)
                        z_S.append(concat_z_S)
                        
                elif scale == self.scales-1: #last scale
                    right_shape_I = (batch_size, 2**((self.dim-1)*(scale-1))*2**self.dim*init_channels)+tuple([x//2**scale for x in init_res])
                    crop_left_shape_I = (batch_size, 2**self.dim*right_shape_I[1])+tuple([x//2 for x in right_shape_I[2:]])
                    z_I.append(T*self.prior.sample(sample_shape=crop_left_shape_I).type_as(D0))

                    right_shape_S = (batch_size, 2**((self.dim-1)*scale)*init_channels)+tuple([x//2**scale for x in init_res])
                    crop_left_shape_S = (batch_size, 2**self.dim*right_shape_S[1])+tuple([x//2 for x in right_shape_S[2:]])
                    concat_z_S = T*self.prior.sample(sample_shape=tuple([scale*crop_left_shape_S[0],])+crop_left_shape_S[1:]).type_as(D0)
                    z_S.append(concat_z_S)
            
            z_short = [z_I, z_S]
        
            if shortcut:
                if temperature_list is not None:
                    z_normal = self.convert_shortcut_to_normal(z_short)
                    z_normal = self.apply_temperature_per_scale(z_normal, temperature_list)
                    z_short = self.convert_normal_to_shortcut(z_normal)  
                
                return z_short
            else:
                z_normal = self.convert_shortcut_to_normal(z_short)
                if temperature_list is not None:
                    z_normal = self.apply_temperature_per_scale(z_normal, temperature_list)
                
                return z_normal

    def apply_temperature_per_scale(self, z_normal, temperature_list):
        assert len(z_normal) == len(temperature_list), 'Temperature list should contain as many elements as the number of scales.'
        num_scales = len(z_normal)
        z_temp_scaled = []
        for i in range(num_scales):
            assert len(z_normal[i])==num_scales-i, 'The order of the latents is reversed.'
            scale_temperature = temperature_list[i]
            z_flow_i = [latent*scale_temperature for latent in z_normal[i]]
            z_temp_scaled.append(z_flow_i)
        
        return z_temp_scaled
        
    def convert_shortcut_to_normal(self, z_short):
        z_I = z_short[0]
        z_S = z_short[1]

        n = len(z_I)
        z = []
        for i in range(n):
            iflow = []
            iflow.append(z_I[i])
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




        

        
        
        
        
        
        