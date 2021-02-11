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
from caflow.data.create_dataset import create_dataset
from pytorch_lightning import Trainer

def main(hparams):
    create_dataset(master_path=hparams.dataroot, resize_size=hparams.load_size, dataset_size=hparams.max_dataset_size)

    train_dataset = TemplateDataset(hparams, phase='train')
    train_dataloader = DataLoader(train_dataset, batch_size=hparams.train_batch,
                                  num_workers=hparams.train_workers)
    
    val_dataset = TemplateDataset(hparams, phase='val')
    val_dataloader = DataLoader(val_dataset, batch_size=hparams.val_batch,
                                num_workers=hparams.val_workers)
    
    trainer = Trainer(gpus=hparams.gpus, accelerator=hparams.accelerator)

    if hparams.checkpoint_path is not None:
        model = CAFlow.load_from_checkpoint(checkpoint_path=hparams.checkpoint_path)
    else:
        model = CAFlow(hparams)

        if hparams.auto_lr_find:
            """It does not lead to improvement of performance. It seems to create a problem in the training dynamics. Do not use it until further inverstigation."""
            # Run learning rate finder
            lr_finder = trainer.tuner.lr_find(model, train_dataloader)
            # Plot with
            #fig = lr_finder.plot(suggest=True)
            #fig.show()
            # Pick point based on plot, or get suggestion
            new_lr = lr_finder.suggestion()
            print('selected learning rate: %.4f' % new_lr)
            # update hparams of the model
            model.hparams.learning_rate = new_lr
    
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    parser = ArgumentParser()
    
    #Trainer arguments
    parser.add_argument('--checkpoint_path', type=str, default=None, help='checkpoint path')
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--accelerator', type=str, default=None, help='automatic pytorch lightning accelerator.')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Accumulates grads every k batches or as set up in the dict.')
    parser.add_argument('--auto_scale_batch_size', type=str, default=None, help='Automatically tries to find the largest batch size that fits into memory, before any training.')
    parser.add_argument('--auto_lr_find', type=bool, default=False, help='Using Lightning’s built-in LR finder.') #do not use it for the time being. Needs to be investigated further.

    #model specific arguments
    parser.add_argument('--data-dim', type=int, default=2)
    parser.add_argument('--data-channels', type=int, default=3)
    parser.add_argument('--model-scales', type=int, default=4)
    parser.add_argument('--model-scale_depth', type=int, default=4)
    parser.add_argument('--train-shortcut', type=bool, default=False)
    
    #The following arguments are used for conditional image sampling in the validation process
    parser.add_argument('--num-val-samples', type=int, default=4, help='num of samples to generate in validation')
    parser.add_argument('--sample-padding', type=int, default=2, help='Amount of padding' )
    parser.add_argument('--sample-normalize', type=bool, default=False, help='If True, shift the image to the range (0, 1), by the min and max values specified by range. Default: False' )
    parser.add_argument('--sample-norm-range', type=tuple, default=None, help='Tuple (min, max) where min and max are numbers, then these numbers are used to normalize the image. \
                                                                        By default, min and max are computed from the tensor.')
    parser.add_argument('--sample-scale-each', type=bool, default=False, help='If True, scale each image in the batch of images separately rather than the (min, max) over all images. Default: False.' )
    parser.add_argument('--sample-pad-value', type=int, default=0)
    
    #program arguments
    parser.add_argument('--dataroot', default='caflow/datasets/edges2shoes', help='path to images')
    parser.add_argument('--train-batch', type=int, default=16, help='train batch size')
    parser.add_argument('--val-batch', type=int, default=4, help='val batch size')
    parser.add_argument('--train-workers', type=int, default=4, help='train_workers')
    parser.add_argument('--val-workers', type=int, default=4, help='val_workers')
    
    #the rest are related to the specific dataset and the required transformations
    parser.add_argument('--max-dataset-size', type=int, default=5000, help='Maximum number of samples allowed per dataset. \
                                                                                Set to float("inf") if you want to use the entire training dataset')
    parser.add_argument('--load-size', type=int, default=64)
    parser.add_argument('--preprocess', type=bool, default=['resize'])
    parser.add_argument('--no-flip', default=True, help='if specified, do not flip the images for data argumentation')
    
    args = parser.parse_args()

    main(args)