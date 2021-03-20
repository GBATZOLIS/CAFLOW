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

        #validation sampling settings
        self.num_val_samples = opts.num_val_u_samples #num of validation samples
        self.sample_padding = opts.sample_padding #bool
        self.sample_normalize = opts.sample_normalize #bool
        self.sample_norm_range = opts.sample_norm_range #tuple range
        self.sample_scale_each = opts.sample_scale_each #bool
        self.sample_pad_value = opts.sample_pad_value #pad value

        self.uflow = UnconditionalFlow(channels=opts.data_channels, dim=opts.data_dim, scales=opts.model_scales, 
                                      scale_depth=opts.rflow_scale_depth, quants=opts.r_quants, vardeq_depth=opts.vardeq_depth)
        
        #set the prior distribution for the latents
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
        #optimiser settings
        self.learning_rate = opts.learning_rate

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
        z = []
        for i in range(1, self.scales+1):
            if i<self.scales:
                sampling_shape = (num_samples, 2**(i*(self.dim-1))*self.channels) + tuple([x//2**i for x in self.resolution])
                z.append(self.prior.sample(sample_shape=sampling_shape).to(self.device))
            elif i==self.scales: #last scale has the same dimensionality as the pre-last just squeezed by a factor of 2.
                sampling_shape = (num_samples, 2**((i-1)*(self.dim-1)+self.dim)*self.channels) + tuple([x//2**i for x in self.resolution])
                z.append(self.prior.sample(sample_shape=sampling_shape).to(self.device))
        
        y, _ = self.uflow(z=z, reverse=True)
        return y
    
    def configure_optimizers(self,):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.999)
        return [optimizer], [scheduler]