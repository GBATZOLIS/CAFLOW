#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:48:40 2021

@author: gbatz97
"""


from argparse import ArgumentParser
#from torch.utils.data.dataloader import default_collate as torch_collate
from torch.utils.data import DataLoader
from caflow.data.aligned_dataset import AlignedDataset
from caflow.data.template_dataset import TemplateDataset
from caflow.models.CAFlow import CAFlow
from caflow.models.UFlow import UFlow
from caflow.data.create_dataset import create_dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

def main(hparams):
    create_dataset(master_path=hparams.dataroot, resize_size=hparams.load_size, dataset_size=hparams.max_dataset_size)

    if hparams.pretrain in ['A', 'B']:
        train_dataset = TemplateDataset(hparams, phase='train', domain=hparams.pretrain)
        train_dataloader = DataLoader(train_dataset, batch_size=hparams.train_batch,
                                      num_workers=hparams.train_workers)
        val_dataset = TemplateDataset(hparams, phase='val', domain=hparams.pretrain)
        val_dataloader = DataLoader(val_dataset, batch_size=hparams.val_batch,
                                    num_workers=hparams.val_workers)
        model = UFlow(hparams)
        trainer = Trainer(num_nodes=hparams.num_nodes, gradient_clip_val=hparams.gradient_clip_val, \
                          gpus=hparams.gpus, accelerator=hparams.accelerator, \
                          accumulate_grad_batches=hparams.accumulate_grad_batches, \
                          resume_from_checkpoint=hparams.resume_from_checkpoint, max_steps=hparams.max_steps, 
                          callbacks=[EarlyStopping('val_loss', patience=100), LearningRateMonitor()])
        trainer.fit(model, train_dataloader, val_dataloader)
        
    else:
        train_dataset = TemplateDataset(hparams, phase='train')
        train_dataloader = DataLoader(train_dataset, batch_size=hparams.train_batch,
                                    num_workers=hparams.train_workers)
        
        val_dataset = TemplateDataset(hparams, phase='val')
        val_dataloader = DataLoader(val_dataset, batch_size=hparams.val_batch,
                                    num_workers=hparams.val_workers)
        
        model = CAFlow(hparams)
        trainer = Trainer(num_nodes=hparams.num_nodes, gpus=hparams.gpus, accelerator=hparams.accelerator, \
                        accumulate_grad_batches=hparams.accumulate_grad_batches, \
                        resume_from_checkpoint=hparams.resume_from_checkpoint, max_steps=hparams.max_steps,
                        callbacks=[EarlyStopping('val_loss', patience=100), LearningRateMonitor()])
        trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    parser = ArgumentParser()

    #pretraining settings
    parser.add_argument('--pretrain', type=str, default='end-to-end', help='which part of the model to pretrain. Default: end-to-end')
    parser.add_argument('--rflow-checkpoint', type=str, default=None)
    parser.add_argument('--tflow-checkpoint', type=str, default=None)
    parser.add_argument('--cflow-checkpoint', type=str, default=None)

    #Trainer arguments
    parser.add_argument('--resume-from-checkpoint', type=str, default=None, help='checkpoint to resume training')
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes. Default=1.')
    
    parser.add_argument('--accelerator', type=str, default=None, help='automatic pytorch lightning accelerator.')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Accumulates grads every k batches or as set up in the dict.')
    parser.add_argument('--gradient_clip_val', type=float, default=0, help='clip the gradient norm computed over all model parameters together')
    # -> Stop criteria
    parser.add_argument('--max-steps', type=int, default=200000) #1

    #optimiser-scheduler settings
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--use-warm-up', type=bool, default=True)
    parser.add_argument('--warm_up', type=int, default=500, help='num of warm up steps.')
    parser.add_argument('--gamma', type=float, default=0.999, help='lr decay factor per epoch')

    #model specific arguments
    parser.add_argument('--data-dim', type=int, default=2)
    parser.add_argument('--data-channels', type=int, default=3)

    parser.add_argument('--shared', default=False, action='store_true', help='Set shared to True if want to use shared conditional flow instead of normal conditional flow.')
    parser.add_argument('--train-shortcut', default=True, type=bool, help='Use the training shortcut if a shared conditional architecture is used.')
    parser.add_argument('--val-shortcut', default=True, type=bool, help='Same as the train-shorcut but for the validation loop.')
    
    parser.add_argument('--model-scales', type=int, default=4)
    parser.add_argument('--rflow-scale-depth', type=int, default=16)
    parser.add_argument('--tflow-scale-depth', type=int, default=16)
    parser.add_argument('--u-cond-scale-depth', type=int, default=4, help='unshared conditional scale depth')
    parser.add_argument('--s-cond-s-scale-depth', type=int, default=4, help='shared conditional shared scale depth')
    parser.add_argument('--s-cond-u-scale-depth', type=int, default=4, help='shared conditional unshared scale depth')
    
    parser.add_argument('--vardeq-depth', type=int, default=0, help='Number of layers in variational dequantisation. If set to None, uniform dequantisation is used.')
    parser.add_argument('--r-quants', type=int, default=256, help='number of quantisation levels of the conditioning image (R in the paper)')
    parser.add_argument('--t-quants', type=int, default=256, help='number of quantisation levels of the conditioned image (T in the paper)')
    
    #Architecture (coupling layer type, NN which parameterises the coupling layer etc.)
    parser.add_argument('--coupling-type', type=str, default='Affine', help='Type of coupling layer. Options=[Affine, MixLog]')
    parser.add_argument('--nn-type', type=str, default='SimpleConvNet', help='nn architecture for the coupling layers. Options=[SimpleConvNet, nnflowpp]')
    ##settings for the SimpleConvNet architecture
    parser.add_argument('--UFLOW-c-hidden-factor', type=int, default=64, help='c_hidden=c_hidden_factor*in_channels')
    parser.add_argument('--CAFLOW-c-hidden-factor', type=int, default=64, help='c_hidden=c_hidden_factor*in_channels')
    ##->settings for the flow++ architecture
    parser.add_argument('--drop-prob', type=float, default=0., help='Dropout probability')
    parser.add_argument('--num-blocks', default=1, type=int, help='Number of blocks in Flow++')
    parser.add_argument('--num-components', default=4, type=int, help='Number of components in the mixture')
    parser.add_argument('--num-channels-factor', default=4, type=int, help='Number of channels in Flow++')
    parser.add_argument('--use-attn', default=False, type=bool, help='whether to use attention in the convolution blocks. Default=False.')
    
    #The following arguments are used for conditional image sampling in the validation process
    parser.add_argument('--num-val-u-samples', type=int, default=64, help='num of samples to generate in validation - unconditional setting')
    parser.add_argument('--num-val-samples', type=int, default=6, help='num of samples to generate in validation - conditional setting')
    parser.add_argument('--sample-padding', type=int, default=2, help='Amount of padding' )
    parser.add_argument('--sample-normalize', default=True, action='store_false', help='If True, shift the image to the range (0, 1), by the min and max values specified by range. Default: False' )
    parser.add_argument('--sample-norm-range', type=tuple, default=(0, 255), help='Tuple (min, max) where min and max are numbers, then these numbers are used to normalize the image. \
                                                                        By default, min and max are computed from the tensor.')
    parser.add_argument('--sample-scale-each', default=False, action='store_true', help='If True, scale each image in the batch of images separately rather than the (min, max) over all images. Default: False.' )
    parser.add_argument('--sample-pad-value', type=int, default=0)
    
    #program arguments
    parser.add_argument('--dataroot', default='caflow/datasets/edges2shoes', help='path to images')
    parser.add_argument('--train-batch', type=int, default=8, help='train batch size')
    parser.add_argument('--val-batch', type=int, default=8, help='val batch size')
    parser.add_argument('--train-workers', type=int, default=4, help='train_workers')
    parser.add_argument('--val-workers', type=int, default=4, help='val_workers')
    
    #the rest are related to the specific dataset and the required transformations
    parser.add_argument('--max-dataset-size', type=int, default=500, help='Maximum number of samples allowed per dataset. \
                                                                                Set to float("inf") if you want to use the entire training dataset')
    parser.add_argument('--load-size', type=int, default=64)
    parser.add_argument('--preprocess', default=['resize'])
    parser.add_argument('--no-flip', default=True, action='store_false', help='if specified, do not flip the images for data argumentation')
    
    args = parser.parse_args()

    main(args)