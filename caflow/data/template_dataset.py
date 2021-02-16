"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from caflow.data.base_dataset import BaseDataset, get_transform
from caflow.data.create_dataset import load_image_paths
import os
import torchvision.transforms as transforms
# from data.image_folder import make_dataset
from PIL import Image
import torch

def discretize(sample):
    return (sample * 255).to(torch.float32)

class TemplateDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    def __init__(self,  opts, phase):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opts)
        
        # get the image paths of your dataset;
        self.image_paths = load_image_paths(opts.dataroot, phase)

        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        #self.transform = get_transform(opts)
        #transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transform_list = [transforms.ToTensor(), discretize]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """

        A_path = self.image_paths['A'][index]
        B_path = self.image_paths['B'][index]
        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        A_transformed = self.transform(A)
        B_transformed = self.transform(B)
        return A_transformed, B_transformed
        
    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths['A'])
