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
    
    

    model = CAFlow(hparams)
    trainer = Trainer(gpus=hparams.gpus)
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    parser = ArgumentParser()
    
    #Trainer arguments
    parser.add_argument('--gpus', default=None)
    
    #model specific arguments
    parser.add_argument('--data-dim', type=int, default=2)
    parser.add_argument('--data-channels', type=int, default=3)
    parser.add_argument('--model-scales', type=int, default=3)
    parser.add_argument('--model-scale_depth', type=int, default=2)
    parser.add_argument('--train-shortcut', type=bool, default=False)
    
    
    #program arguments
    parser.add_argument('--dataroot', default='caflow/datasets/edges2shoes', help='path to images')
    parser.add_argument('--train-batch', type=int, default=4, help='train batch size')
    parser.add_argument('--val-batch', type=int, default=4, help='val batch size')
    parser.add_argument('--train-workers', type=int, default=4, help='train_workers')
    parser.add_argument('--val-workers', type=int, default=4, help='val_workers')
    #the rest are related to the specific dataset and the required transformations
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset')
    parser.add_argument('--load-size', type=int, default=64)
    parser.add_argument('--preprocess', type=bool, default=['resize'])
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
    
    args = parser.parse_args()

    main(args)