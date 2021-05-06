from caflow.data.create_dataset import create_dataset
from caflow.data.template_dataset import TemplateDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from caflow.models.CAFlow import CAFlow
import os
import glob
from argparse import ArgumentParser
import torchvision
import torch
from tqdm import tqdm
from caflow.models.UFlow import UFlow
import numpy as np

def draw_samples(writer, model, Y, I, num_samples, temperature_list, num_batch):
    B = Y.shape[0]
    raw_length = 1+num_samples+1
    all_images = torch.zeros(tuple([B*raw_length,]) + I.shape[1:], device=torch.device('cpu'), requires_grad=False)
    
    for i in range(B):
        all_images[i*raw_length] = Y[i]
        all_images[(i+1)*raw_length-1] = I[i]
            
    # generate images
    for j in range(1, num_samples+1):
        sampled_image = model.sample(Y, shortcut=model.val_shortcut, temperature_list=temperature_list).detach().cpu()
        for i in range(B):
            all_images[i*raw_length+j]=sampled_image[i]
                
    grid = torchvision.utils.make_grid(
                tensor=all_images,
                nrow = raw_length, #Number of images displayed in each row of the grid
                padding=model.sample_padding,
                normalize=model.sample_normalize,
                range=model.sample_norm_range,
                scale_each=model.sample_scale_each,
                pad_value=model.sample_pad_value,
        )
    temp_string = ['%.2f' % x for x in temperature_list]
    temp_string='_'.join(temp_string)
    str_title = 'valbatch_%d_epoch_%d_' % (num_batch, model.current_epoch)+temp_string
    writer.add_image(str_title, grid)
    writer.flush()

def annealed_distribution_uflow(hparams):
    device = torch.device('cuda:%s' % hparams.gpus) if hparams.gpus is not None else torch.device('cpu')
    log_dir = os.path.join('lightning_logs','version_%d' % hparams.experiment)
    model = UFlow.load_from_checkpoint(checkpoint_path=glob.glob(os.path.join(log_dir, 'checkpoints', '*.ckpt'))[0]).to(device)
    model.eval()

    annealed_samples = model.sample_from_annealed_distribution(num_samples=hparams.num_samples, T=hparams.T, burn=hparams.burn)
    print(annealed_samples.size()) 

    writer = SummaryWriter(log_dir=log_dir, comment='annealed_distribution_samples')
    grid = torchvision.utils.make_grid(
                tensor = annealed_samples,
                nrow = int(np.sqrt(annealed_samples.size(0))), #Number of images displayed in each row of the grid
                padding=model.sample_padding,
                normalize=model.sample_normalize,
                range=model.sample_norm_range,
                scale_each=model.sample_scale_each,
                pad_value=model.sample_pad_value,
            )
    
    str_title = 'annealed_distribution_samples_T_%.3f' % hparams.T
    writer.add_image(str_title, grid)
    writer.flush()
    writer.close()

def main(hparams):
    create_dataset(master_path=hparams.dataroot, resize_size=hparams.load_size, dataset_size=hparams.max_dataset_size)
    val_dataset = TemplateDataset(hparams, phase='val')
    val_dataloader = DataLoader(val_dataset, batch_size=hparams.val_batch,
                                    num_workers=hparams.val_workers)

    device = torch.device('cuda:%s' % hparams.gpus) if hparams.gpus is not None else torch.device('cpu')

    log_dir = os.path.join('lightning_logs','version_%d' % hparams.experiment)
    model = CAFlow.load_from_checkpoint(checkpoint_path=glob.glob(os.path.join(log_dir, 'checkpoints', '*.ckpt'))[0]).to(device)
    model.eval()

    writer = SummaryWriter(log_dir=log_dir, comment='testing')

    '''
    temperature_lists = [[1, 1, 1, 1],
                         [1, 1, 1, 0.75],
                         [1, 1, 0.75, 0.75],
                         [1, 0.75, 0.75, 0.75],
                         [0.75, 0.75, 0.75, 0.75],
                         [1, 0.8, 0.6, 0.4],
                         [0.4, 0.6, 0.8, 1],
                         [1, 1, 1, 1],
                         [0.75, 1, 1, 1],
                         [0.75, 0.75, 1, 1],
                         [0.75, 0.75, 0.75, 1],
                         [0.75, 0.75, 0.75, 0.75],
                         [0.5, 0.5, 0.5, 0.5],
                         [0.25, 0.25, 0.25, 0.25],
                         [0.1, 0.1, 0.1, 0.1],
                         [0.01, 0.01, 0.01, 0.01],
                         [0.65, 0.7, 0.75, 0.8],
                         [0.70, 0.75, 0.8, 0.85],
                         [0.75, 0.8, 0.85, 0.9],
                         [0.8, 0.85, 0.9, 0.95]
                         ]   
    '''

    temperature_lists = [[0.8, 0.8, 0.8, 0.8],
                         [0.9, 0.9, 0.9, 0.9],
                         [0.97, 0.97, 0.97, 0.97]
                         ]   

    for temperature_list in tqdm(temperature_lists):
        for step, (x,y) in tqdm(enumerate(val_dataloader)):
            x, y = x.to(device), y.to(device)
            #if step > 0:
            #    continue
            draw_samples(writer, model, x, y, hparams.num_samples, temperature_list, step)
    
    writer.close()

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--gpus', default=None)
    #program arguments
    parser.add_argument('--dataroot', default='caflow/datasets/edges2shoes', help='path to images')
    parser.add_argument('--val-batch', type=int, default=8, help='val batch size')
    parser.add_argument('--val-workers', type=int, default=4, help='val_workers')
    #the rest are related to the specific dataset and the required transformations
    parser.add_argument('--max-dataset-size', type=int, default=40000, help='Maximum number of samples allowed per dataset. \
                                                                                Set to float("inf") if you want to use the entire training dataset')
    parser.add_argument('--load-size', type=int, default=64)

    # Testing
    parser.add_argument('--experiment', type=int, default=0, help='which experiment to test.')
    parser.add_argument('--num-samples', type=int, default=64, help='num of samples to generate in testing.')
    parser.add_argument('--T', type=float, default=0.97, help='Annealing temperature.')
    parser.add_argument('--burn', type=int, default=500, help='Burn-in time in NUTS algorithm.')

    args = parser.parse_args()

    main(args)
    #annealed_distribution_uflow(args)
    
