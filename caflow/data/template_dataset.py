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
import numpy as np
import scipy

def discretize(sample):
    return (sample * 255).to(torch.float32)

class TemplateDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    def __init__(self,  opts, phase, domain=None):
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
        
        #set domain to the domain you want to sample from. If domain is set to N we get paired samples from all domains.
        self.domain = domain

        # get the image paths of your dataset;
        self.image_paths = load_image_paths(opts.dataroot, phase)
        _, file_extension = os.path.splitext(self.image_paths['A'][0])
        self.file_extension = file_extension

        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        if self.file_extension in ['.jpg', '.png']:
            transform_list = [transforms.ToTensor(), discretize]
        elif self.file_extension in ['.npy']:
            self.channels = opts.data_channels
            self.resolution = opts.resolution
            transform_list = [torch.from_numpy, lambda x: x.type(torch.FloatTensor)]
        else:
            raise Exception('File extension %s is not supported yet. Please update the code.' % self.file_extension)

        self.transform = transforms.Compose(transform_list)


    def __getitem__(self, index):
        if self.domain is None:
            A_path = self.image_paths['A'][index]
            B_path = self.image_paths['B'][index]

            #load the paired images/scans
            if self.file_extension in ['.jpg', '.png']:
                A = Image.open(A_path).convert('RGB')
                B = Image.open(B_path).convert('RGB')
            elif self.file_extension in ['.npy']:
                A = np.load(A_path)
                B = np.load(B_path)
                
                #reshape/slice appropriately
                if self.channels == 1:
                    #slicing
                    def get_starting_index(A, resolution, axis):
                        if A.shape[axis] == self.resolution[axis]:
                            starting_index = 0
                        elif A.shape[axis] > self.resolution[axis]:
                            starting_index = np.random.randint(0, A.shape[axis]-self.resolution[axis])
                        else:
                            raise Exception('requested resolution exceeds data resolution in axis %d' % axis)
                        return starting_index

                    #i0, i1, i2 = get_starting_index(A, self.resolution, 0), get_starting_index(A, self.resolution, 1), get_starting_index(A, self.resolution, 2)
                    
                    #i0, i1, i2 = 0, 0, 20
                    #A = A[i0:i0+self.resolution[0], i1:i1+self.resolution[1], i2:i2+self.resolution[2]]
                    #B = B[i0:i0+self.resolution[0], i1:i1+self.resolution[1], i2:i2+self.resolution[2]]

                    #------rotation-------
                    #angle = [0, 90, 180, 270][np.random.randint(4)]
                    #axes_combo = [(0, 1), (1, 2), (0, 2)][np.random.randint(3)]
                    #if angle != 0:
                    #    A = scipy.ndimage.rotate(A, angle=angle, axes=axes_combo)
                    #    B = scipy.ndimage.rotate(B, angle=angle, axes=axes_combo)

                    #dequantise 0 value
                    #A[A==0.]=10**(-6)*np.random.rand()
                    #B[B==0.]=10**(-6)*np.random.rand()
                    
                    #expand dimensions to acquire a pytorch-like form.
                    A = np.expand_dims(A, axis=0)
                    B = np.expand_dims(B, axis=0)

                elif self.channels > 1 and self.channels < A.shape[-1]:
                    starting_slicing_index = np.random.randint(0, A.shape[-1] - self.channels)
                    A = A[:,:,starting_slicing_index:starting_slicing_index+self.channels]
                    B = B[:,:,starting_slicing_index:starting_slicing_index+self.channels]

                    A = np.moveaxis(A, -1, 0)
                    B = np.moveaxis(B, -1, 0)

                elif self.channels == A.shape[-1]:
                    A = np.moveaxis(A, -1, 0)
                    B = np.moveaxis(B, -1, 0)
                else:
                    raise Exception('Invalid number of channels.')

            else:
                raise Exception('File extension %s is not supported yet. Please update the code.' % self.file_extension)
            
            #transform the images/scans
            A_transformed = self.transform(A)
            B_transformed = self.transform(B)

            return A_transformed, B_transformed
        else:
            path = self.image_paths[self.domain][index]

            #load the image/scan
            if self.file_extension in ['.jpg', '.png']:
                img = Image.open(path).convert('RGB')
            elif self.file_extension in ['.npy']:
                img = np.load(path)

                #dequantise 0 value
                #img[img<10**(-6)]=10**(-6)*np.random.rand()

                #reshape/slice appropriately
                if self.channels == 1:
                    img = np.expand_dims(img, axis=0)
                elif self.channels > 1 and self.channels < img.shape[-1]:
                    starting_slicing_index = np.random.randint(0, img.shape[-1] - self.channels)
                    img = img[:,:,starting_slicing_index:starting_slicing_index+self.channels]
                    img = np.moveaxis(img, -1, 0)
                elif self.channels == img.shape[-1]:
                    img = np.moveaxis(img, -1, 0)
                else:
                    raise Exception('Invalid number of channels.')

            #transform the image/scan
            img_transformed = self.transform(img)
            
            return img_transformed
        
    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths['A'])
