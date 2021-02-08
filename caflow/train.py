#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:48:40 2021

@author: gbatz97
"""


from argparse import ArgumentParser
from torch.utils.data.dataloader import default_collate as torch_collate
from caflow.data.DataLoader import DataLoader
from caflow.data.aligned_dataset import AlignedDataset
from caflow.models.CAFlow import CAFlow
from caflow.utils.TensorboardImageSampler import TensorboardConditionalImageSampler
from pytorch_lightning import Trainer

def main(hparams):
    train_dataset = AlignedDataset(hparams, phase='train')
    train_dataloader = DataLoader(train_dataset, batch_size=hparams.train_batch,
                                  num_workers=hparams.train_workers, 
                                  collate_fn=torch_collate)
    
    val_dataset = AlignedDataset(hparams, phase='val')
    val_dataloader = DataLoader(val_dataset, batch_size=hparams.val_batch,
                                num_workers=hparams.val_workers, 
                                collate_fn=torch_collate)
    
    if hparams.checkpoint_path is not None:
        model = CAFlow.load_from_checkpoint(checkpoint_path=hparams.checkpoint_path)
    else:
        model = CAFlow(hparams)
        
    trainer = Trainer(gpus=hparams.gpus)
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    parser = ArgumentParser()
    
    #Trainer arguments
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='checkpoint path')
    
    #model specific arguments
    parser.add_argument('--data-dim', type=int, default=2)
    parser.add_argument('--data-channels', type=int, default=3)
    parser.add_argument('--model-scales', type=int, default=3)
    parser.add_argument('--model-scale_depth', type=int, default=8)
    parser.add_argument('--train-shortcut', type=bool, default=False)
    
    #The following arguments are used for conditional image sampling in the validation process
    parser.add_argument('--num_val_samples', type=int, default=3, help='num of samples to generate in validation')
    parser.add_argument('--sample_padding', type=int, default=2, help='Amount of padding' )
    parser.add_argument('--sample_normalize', type=bool, default=False, help='If True, shift the image to the range (0, 1), by the min and max values specified by range. Default: False' )
    parser.add_argument('--sample_norm_range', type=tuple, default=None, help='Tuple (min, max) where min and max are numbers, then these numbers are used to normalize the image. \
                                                                        By default, min and max are computed from the tensor.')
    parser.add_argument('--sample_scale_each', type=bool, default=False, help='If True, scale each image in the batch of images separately rather than the (min, max) over all images. Default: False.' )
    parser.add_argument('--sample_pad_value', type=int, default=0)
    
    #program arguments
    parser.add_argument('--dataroot', default='caflow/datasets/edges2shoes', help='path to images')
    parser.add_argument('--train-batch', type=int, default=4, help='train batch size')
    parser.add_argument('--val-batch', type=int, default=4, help='val batch size')
    parser.add_argument('--train-workers', type=int, default=4, help='train_workers')
    parser.add_argument('--val-workers', type=int, default=4, help='val_workers')
    #the rest are related to the specific dataset and the required transformations
    parser.add_argument('--max_dataset_size', type=int, default=500, help='Maximum number of samples allowed per dataset. \
                                                                                Set to float("inf") if you want to use the entire training dataset')
    parser.add_argument('--load-size', type=int, default=32)
    parser.add_argument('--preprocess', type=bool, default=['resize'])
    parser.add_argument('--no_flip', default=True, help='if specified, do not flip the images for data argumentation')
    
    args = parser.parse_args()

    main(args)