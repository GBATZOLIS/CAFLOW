from caflow.models.modules.mri_to_pet.UnconditionalFlow import UnconditionalFlow
from caflow.models.modules.mri_to_pet.SimpleConditionalFlow import SimpleConditionalFlow
from caflow.models.UFlow import UFlow
from caflow.utils.processing import normalise
import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision
import numpy as np

class DualGlow(pl.LightningModule):
    def __init__(self, opts):
        super(DualGlow, self).__init__()
        self.save_hyperparameters()

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

        self.model['rflow'] = UFlow(dim=opts.data_dim, scales=opts.model_scales, channels=opts.data_channels, 
                                        resolution=opts.resolution, scale_depth=opts.rflow_scale_depth,
                                        use_inv_scaling=opts.use_inv_scaling, scaling_range=opts.r_domain_range, 
                                        use_dequantisation=opts.use_dequantisation, quants=opts.r_quants, 
                                        vardeq_depth=opts.vardeq_depth, opts=opts)
        
        self.model['tflow'] = UFlow(dim=opts.data_dim, scales=opts.model_scales, channels=opts.data_channels, 
                                        resolution=opts.resolution, scale_depth=opts.tflow_scale_depth,
                                        use_inv_scaling=opts.use_inv_scaling, scaling_range=opts.t_domain_range, 
                                        use_dequantisation=opts.use_dequantisation, quants=opts.t_quants, 
                                        vardeq_depth=opts.vardeq_depth, opts=opts)
        

        ### we need to create SimpleConditionalFlow and import it in this file.
        self.model['ConditionalFlow'] = SimpleConditionalFlow(channels=opts.data_channels, 
                                                                dim=opts.data_dim, 
                                                                resolution = self.resolution,
                                                                scales=opts.model_scales, 
                                                                scale_depth=opts.u_cond_scale_depth,
                                                                nn_settings=self.create_nn_settings(opts))
        

        #set the prior distribution for the latents
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

        #loss function settings
        self.lamda = opts.lamda

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

    def forward(self, Y):
        return self.sample(Y)
    
    def logjoint(self, Y, I, scaled=True):
        D, rlogprior, rlogdet = self.model['rflow'](y=Y)
        L, _, tlogdet = self.model['tflow'](y=I)
        
        Z_cond, condlogprior, condlogdet = self.model['ConditionalFlow'](L=L, z=[], D=D, reverse=False)

        logjoint = self.lamda*(rlogprior + rlogdet) + tlogdet + condlogprior + condlogdet

        if scaled:
            scaled_logjoint = logjoint/(np.prod(Y.shape[1:])*np.prod(I.shape[1:]))
            return scaled_logjoint, [D, Z_cond, L]
        else:
            return logjoint, [D, Z_cond, L]

    def training_step(self, batch, batch_idx):
        Y, I = batch
        scaled_logjoint, encodings = self.logjoint(Y, I, scaled=True)
        loss = -1*torch.mean(scaled_logjoint)
        
        #logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def training_step_end(self, training_step_outputs):
        return training_step_outputs.sum()
    
    def validation_step(self, batch, batch_idx):
        Y, I = batch
        scaled_logjoint, encodings = self.logjoint(Y, I, scaled=True)
        loss = -1*torch.mean(scaled_logjoint)
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        
        I_sample =  self.sample(Y)
        val_rec_loss = torch.mean(torch.abs(I-I_sample))
        self.log('val_rec_loss', val_rec_loss, on_step=True, on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            B = Y.shape[0]
            raw_length = 1+self.num_val_samples+1
            all_images = torch.zeros(tuple([B*raw_length,]) + I.shape[1:])
                    
            for i in range(B):
                all_images[i*raw_length] = Y[i]
                all_images[(i+1)*raw_length-1] = I[i]
                    
            for sampling_T in self.sampling_temperatures:
                # generate images
                with torch.no_grad():
                    for j in range(1, self.num_val_samples+1):
                        sampled_image = self.sample(Y, T=sampling_T)
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
    
    def sample(self, Y, T=1):
        D, _, _ =self.model['rflow'](y=Y) #D = [D_(n-1), D_(n-2), ..., D_2, D_1, D_0]

        z_normal = self.generate_z_cond(D, T=T)
        L_pred, _ = self.model['ConditionalFlow'](L=[], z=z_normal, D=D, reverse=True)
        I_sample, _ = self.model['tflow'](z=L_pred, reverse=True)
        return I_sample
    
    def generate_z_cond(self, D, T=1):
        z=[]
        for i in range(len(D)):
            z.append(T*self.prior.sample(sample_shape=D[i].shape).type_as(D[i]))
        return z