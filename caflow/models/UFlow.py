from caflow.models.modules.mri_to_pet.UnconditionalFlow import UnconditionalFlow
import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision
import numpy as np

class UFlow(pl.LightningModule):
    def __init__(self, opts):
        super(UFlow, self).__init__()
        self.save_hyperparameters()
        self.dim = opts.data_dim
        self.scales = opts.model_scales
        self.channels = opts.data_channels
        self.resolution = [opts.load_size for _ in range(self.dim)]
        self.total_dims = self.channels*np.prod(self.resolution) #-->addition

        #validation sampling settings
        self.num_val_samples = opts.num_val_u_samples #num of validation samples
        self.sample_padding = opts.sample_padding #bool
        self.sample_normalize = opts.sample_normalize #bool
        self.sample_norm_range = opts.sample_norm_range #tuple range
        self.sample_scale_each = opts.sample_scale_each #bool
        self.sample_pad_value = opts.sample_pad_value #pad value

        self.uflow = UnconditionalFlow(channels=opts.data_channels, dim=opts.data_dim, resolution=opts.load_size, scales=opts.model_scales, 
                                      scale_depth=opts.rflow_scale_depth, quants=opts.r_quants, vardeq_depth=opts.vardeq_depth, coupling_type=opts.coupling_type,
                                      nn_settings=self.create_nn_settings(opts))
        
        #set the prior distribution for the latents
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
        #optimiser settings
        self.learning_rate = opts.learning_rate
        self.use_warm_up = opts.use_warm_up
        self.warm_up = opts.warm_up
        self.gamma = opts.gamma

    def create_nn_settings(self, opts):
        nn_settings={'nn_type':opts.nn_type, 'c_hidden_factor':opts.UFLOW_c_hidden_factor, \
            'drop_prob':opts.drop_prob, 'num_blocks':opts.num_blocks, 'use_attn':opts.use_attn,\
            'num_components':opts.num_components, 'num_channels_factor':opts.num_channels_factor}
        return nn_settings

    def forward(self, y=None, z=[], logprior=0., logdet=0., reverse=False):
        return self.uflow(y=y, z=z, logprior=logprior, logdet=logdet, reverse=reverse)

    def logprob(self, Y, scaled=True):
        _, logprior, logdet = self.uflow(y=Y)
        logprob = logprior + logdet
        if scaled:
            scaled_logprob = logprob*np.log2(np.exp(1)) / np.prod(Y.shape[1:])
            return scaled_logprob
        else:
            return logprob

    def training_step(self, batch, batch_idx):
        Y = batch
        scaled_logprob = self.logprob(Y, scaled=True)
        loss = -1*torch.mean(scaled_logprob)

        #logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def training_step_end(self, training_step_outputs):
        return training_step_outputs.sum()
    
    def validation_step(self, batch, batch_idx):
        Y = batch
        scaled_logprob = self.logprob(Y, scaled=True)
        loss = -1*torch.mean(scaled_logprob)

        #logging
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def validation_epoch_end(self, outputs):
        y = self.sample(num_samples = self.num_val_samples)
        grid = torchvision.utils.make_grid(
                tensor = y,
                nrow = int(np.sqrt(self.num_val_samples)), #Number of images displayed in each row of the grid
                padding=self.sample_padding,
                normalize=self.sample_normalize,
                range=self.sample_norm_range,
                scale_each=self.sample_scale_each,
                pad_value=self.sample_pad_value,
            )

        str_title = 'val_samples_epoch_%d' % self.current_epoch
        self.logger.experiment.add_image(str_title, grid, self.current_epoch)

    def sample(self, num_samples):
        flattened_latents = self.prior.sample(sample_shape=(num_samples, self.total_dims)).to(self.device)
        scale_z = self.convert_to_scale_tensor(flattened_latents)
        y, _ = self.uflow(z=scale_z, reverse=True)
        return y

    def sample_from_annealed_distribution(self, T=0.97, num_samples=1):
        def log_prob(z):
            assert z.size(0)==1, 'z must have shape of the form (1,...)'
            scale_tensor = self.convert_to_scale_tensor(z)
            y, logdet = self.uflow(z=scale_tensor, reverse=True)
            return gamma*self.prior.log_prob(z).sum()+(1-gamma)*logdet.sum()

        gamma = 1.0/(T**2.0)
        burn=500
        step_size = .3
        L = 5
        N_nuts = burn + 1
        for _ in range(num_samples):
            params_init = self.prior.sample(self.total_dims)
            params_hmc_nuts = hamiltorch.sample(log_prob_func=log_prob, params_init=params_init,
                                                num_samples=N_nuts,step_size=step_size,num_steps_per_sample=L,
                                                sampler=hamiltorch.Sampler.HMC_NUTS, burn=burn,
                                                desired_accept_rate=0.8)
    
    def convert_to_scale_tensor(self, flattened_tensor):
        #input: flattened_tensor
        #output: z reshaped in a scale tensor - a list of number_scales tensors - [(batch, 2channels, res/2, res/2), (batch, 4channels, res/4, res/4)...]
        batch_size = flattened_tensor.size(0)
        previous_dimensions=0
        scale_tensor = []
        for i in range(1, self.scales+1):
            if i<self.scales:
                scale_shape = (2**(i*(self.dim-1))*self.channels, ) + tuple([x//2**i for x in self.resolution])
            elif i==self.scales: #last scale has the same dimensionality as the pre-last just squeezed by a factor of 2.
                scale_shape = (2**((i-1)*(self.dim-1)+self.dim)*self.channels, )+tuple([x//2**i for x in self.resolution])

            new_dimensions = np.prod(scale_shape)
            scale = torch.reshape(flattened_tensor[:, previous_dimensions:new_dimensions], shape=(batch_size,) + scale_shape)
            scale_tensor.append(scale)
        return scale_tensor
    
    def convert_to_flattened_tensor(self, scale_tensor):
        #input: scale tensor
        #output: flattened tensor
        flattened_tensor = []
        for i in range(1, self.scales+1):
            flattened_tensor.append(torch.reshape(scale_tensor[i], (scale_tensor[i].size(0), -1)))
        flattened_tensor = torch.cat(flattened_tensor, dim=1)
        return flattened_tensor



    
    def configure_optimizers(self,):
        def scheduler_lambda_function(s):
            #warmup until it reaches scale 1 and then STEP LR decrease every other epoch with gamma factor.
            if self.use_warm_up:
                if s < self.warm_up:
                    return s / self.warm_up
                else:
                    return self.gamma**(self.current_epoch)
            else:
                return self.gamma**(self.current_epoch)
        
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {'scheduler': optim.lr_scheduler.LambdaLR(optimizer, scheduler_lambda_function),
                    'interval': 'step'}  # called after each training step

        #lambda s: min(1., s / self.warm_up) -> warm_up lambda
        return [optimizer], [scheduler]
    

    @property
    def num_training_steps_per_epoch(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)     

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return batches // effective_accum