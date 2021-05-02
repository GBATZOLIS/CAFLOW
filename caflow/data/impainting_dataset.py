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

        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        #self.transform = get_transform(opts)
        #transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transform_list = [transforms.ToTensor(), discretize]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        if self.domain is None:
            A_path = self.image_paths['A'][index]
            B_path = self.image_paths['B'][index]
            A = Image.open(A_path).convert('RGB')
            B = Image.open(B_path).convert('RGB')
            A_transformed = self.transform(A)
            B_transformed = self.transform(B)
            return A_transformed, B_transformed
        else:
            path = self.image_paths[self.domain][index]
            img = Image.open(path).convert('RGB')
            img_transformed = self.transform(img)
            return img_transformed
        
    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths['A'])