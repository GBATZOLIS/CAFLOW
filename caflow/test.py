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
from pathlib import Path
from piqa.psnr import psnr, mse
from piqa.lpips import LPIPS
from torchvision.utils import save_image

def annealed_distribution_uflow(hparams):
    device = torch.device('cuda:%s' % hparams.gpu) if hparams.gpu is not None else torch.device('cpu')
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

def add_jitter(z_cond_unshared, std):
    noise_distribution = torch.distributions.normal.Normal(loc=0.0, scale=std)
    z_jittered = []
    for scale_z_latents in z_cond_unshared:
        jittered_scale = []
        for latent in scale_z_latents:
            jitter = noise_distribution.sample(sample_shape=latent.shape).type_as(latent)
            jittered_latent = jitter + latent
            jittered_scale.append(jittered_latent)
        z_jittered.append(jittered_scale)
    return z_jittered

def pertub_all_but_last_scale(z_cond_unshared, std):
    noise_distribution = torch.distributions.normal.Normal(loc=0.0, scale=std)
    z_jittered = []
    for scale_z_latents in z_cond_unshared:
        if len(scale_z_latents) == 1:
            z_jittered.append(scale_z_latents)
        else:
            jittered_scale = []
            for latent in scale_z_latents:
                jitter = noise_distribution.sample(sample_shape=latent.shape).type_as(latent)
                jittered_latent = jitter
                jittered_scale.append(jittered_latent)
            z_jittered.append(jittered_scale)
    return z_jittered


def conditional_log_prob(Y, I, model):
    D, _, _ = model.model['rflow'](y=Y)
    L, _, tlogdet = model.model['tflow'](y=I)
    Z_cond, condlogprior, condlogdet = model.model['UnsharedConditionalFlow'](L=L, z=[], D=D, reverse=False)
    cond_log_prob = (tlogdet + condlogprior + condlogdet).detach().cpu()
    return cond_log_prob


'''
    #generated jittered images
    jitter=75e-2
    jittered_images = all_images.clone()
    for j in tqdm(range(1, num_samples+1)):
        # Sampling
        #z_jittered = add_jitter(Z_cond, std=jitter)
        z_jittered = pertub_all_but_last_scale(Z_cond, std=jitter)
        L_pred, _ = model.model['UnsharedConditionalFlow'](L=[], z=z_jittered, D=D, reverse=True)
        sampled_image, _ = model.model['tflow'](z=L_pred, reverse=True)
        sampled_image = sampled_image.detach().cpu()
        for i in range(B):
            jittered_images[i*raw_length+j] = sampled_image[i]

    #plot jittered images
    grid = torchvision.utils.make_grid(
                tensor=jittered_images,
                nrow = raw_length, #Number of images displayed in each row of the grid
                padding=model.sample_padding,
                normalize=model.sample_normalize,
                range=model.sample_norm_range,
                scale_each=model.sample_scale_each,
                pad_value=model.sample_pad_value,
            )

    str_title = 'valbatch_%d_epoch_%d_jitter_%.8f' % (batch_ID, model.current_epoch, jitter)
    writer.add_image(str_title, grid)
    writer.flush()
'''


def plot_samples(writer, model, all_images, title, raw_length):
    #Standard plotting procedure
    grid = torchvision.utils.make_grid(
                tensor=all_images,
                nrow = raw_length, #Number of images displayed in each row of the grid
                padding=model.sample_padding,
                normalize=model.sample_normalize,
                range=model.sample_norm_range,
                scale_each=model.sample_scale_each,
                pad_value=model.sample_pad_value,
            )

    writer.add_image(title, grid)
    writer.flush()


def draw_samples(writer, model, Y, I, num_samples, temperature_list, batch_ID, plot, num_selected_samples):
    B = Y.shape[0]
    raw_length = 1+num_samples+1
    all_images = torch.zeros(tuple([B*raw_length,]) + I.shape[1:], device=torch.device('cpu'), requires_grad=False)
    cond_log_probs = torch.zeros(B*raw_length, device=torch.device('cpu'), requires_grad=False)

    for i in range(B):
        all_images[i*raw_length] = Y[i]
        all_images[(i+1)*raw_length-1] = I[i]

    #Calculate the conditional log probabilities of the ground truth images
    D, _, _ = model.model['rflow'](y=Y)
    L, _, tlogdet = model.model['tflow'](y=I)
    Z_cond, condlogprior, condlogdet = model.model['UnsharedConditionalFlow'](L=L, z=[], D=D, reverse=False)
    cond_log_prob = (tlogdet + condlogprior + condlogdet).detach().cpu()
    for i in range(B):
        cond_log_probs[(i+1)*raw_length-1] = cond_log_prob[i]

    # generate images
    for j in range(1, num_samples+1):
        #decoding
        sampled_image = model.sample(Y, shortcut=model.val_shortcut, temperature_list=temperature_list)
        
        #encoding
        cond_log_prob = conditional_log_prob(Y, sampled_image, model)
        for i in range(B):
            all_images[i*raw_length+j] = sampled_image[i].detach().cpu()
            cond_log_probs[i*raw_length+j] = cond_log_prob[i]

    if plot:
        temp_string = ['%.2f' % x for x in temperature_list]
        temp_string='_'.join(temp_string)
        title = 'valbatch_%d_epoch_%d_' % (batch_ID, model.current_epoch)+temp_string
        plot_samples(writer, model, all_images, title, raw_length)

    #Plot the images in descending conditional log probability. j: 1, num_samples+1
    reordered_all_images = all_images.clone()
    include_gt = 0
    for i in range(B):
        cond_samples = all_images[i*raw_length+1:i*raw_length+num_samples+1+include_gt].clone()
        cond_log_probabilities = cond_log_probs[i*raw_length+1:i*raw_length+num_samples+1+include_gt]

        sorted_indices = torch.argsort(cond_log_probabilities, dim=-1, descending=True)
        reordered_all_images[i*raw_length+1:i*raw_length+num_samples+1+include_gt] = cond_samples[sorted_indices, ::]
    
    if plot:
        temp_string = ['%.2f' % x for x in temperature_list]
        temp_string='_'.join(temp_string)
        title = 'valbatch_%d_epoch_%d_descending_cond_log_prob' % (batch_ID, model.current_epoch)+temp_string
        plot_samples(writer, model, reordered_all_images, title, raw_length)

    selected_images = []
    for i in range(B):
        selected = reordered_all_images[i*raw_length+1:i*raw_length+num_selected_samples+1]
        selected_images.append(selected)
    selected_images = torch.stack(selected_images)

    return selected_images

def calculate_pixel_std(samples):
    #samples.size = (batch, samples, channels, height, width)
    return torch.mean(torch.std(samples, dim=1, unbiased=True, keepdim=False))

def main(hparams):
    if hparams.create_dataset:
        create_dataset(master_path=hparams.dataroot, resize_size=hparams.load_size, \
                       dataset_size=hparams.max_dataset_size, dataset_style=hparams.dataset_style, \
                       mask_to_area=hparams.mask_to_area, mask_type=hparams.mask_type,
                       sr_factor=hparams.sr_factor)

    val_dataset = TemplateDataset(hparams, phase='val')
    val_dataloader = DataLoader(val_dataset, batch_size=hparams.val_batch,
                                    num_workers=hparams.val_workers)

    device = torch.device('cuda:%s' % hparams.gpu) if hparams.gpu is not None else torch.device('cpu')

    base_dir = os.path.join('lightning_logs','version_%d' % hparams.experiment)

    model = CAFlow.load_from_checkpoint(checkpoint_path=glob.glob(os.path.join(base_dir, 'checkpoints', '*.ckpt'))[-1]).to(device)
    model.eval()

    writer_dir = os.path.join(base_dir, 'testing')
    Path(writer_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=writer_dir, comment='testing')

    images_dir = os.path.join(base_dir, 'images')
    Path(images_dir).mkdir(parents=True, exist_ok=True)

    average_rmse = []
    lpips = LPIPS(scaling=False, reduction='none').to(device)
    average_lpips = []
    average_pixel_stds = []
    with torch.no_grad():
        temperature_lists = [[T for _ in range(model.scales)] for T in hparams.temperatures]
        for temperature_list in temperature_lists:
            for step, (x,y) in tqdm(enumerate(val_dataloader)):
                #if step > 0:
                #    break

                x, y = x.to(device), y.to(device)
                selected_samples = draw_samples(writer, model, x, y, hparams.num_samples, temperature_list, step, hparams.plot, hparams.num_selected_samples)

                if hparams.num_selected_samples == 1:
                    selected_samples = torch.squeeze(selected_samples, dim=1)
                    for j in range(selected_samples.size(0)):
                        save_image(selected_samples[j], os.path.join(images_dir, 'img_%d_%d.png'%(step, j)))
                    average_rmse.append(torch.mean(torch.sqrt(mse(selected_samples.to(device), y))).item())
                    average_lpips.append(torch.mean(lpips(selected_samples.to(device)/255, y/255)).item())
                else:
                    #1.) calculate the standard deviation of the pixels
                    avg_pixel_std = calculate_pixel_std(selected_samples)
                    average_pixel_stds.append(avg_pixel_std)

                    average_rmse.append(torch.mean(torch.sqrt(mse(selected_samples[:,0,::].to(device), y))).item())
                    average_lpips.append(torch.mean(lpips(selected_samples[:,0,::].to(device)/255, y/255)).item())

                    #2.) save images
                    for j in range(selected_samples.size(0)):
                        for i in range(selected_samples.size(1)):
                            save_image(selected_samples[j][i], os.path.join(images_dir, 'img_%d_%d_%d.png'%(step, j, i)), normalize = True)


                

    print('Average RMSE: %.2f' % np.mean(average_rmse))
    print('Average LPIPS: %.2f' % np.mean(average_lpips))
    print('Average pixel std: %.3f' % np.mean(average_pixel_stds))

    writer.close()

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--gpu', type=str, default=None)

    #program arguments
    parser.add_argument('--dataroot', default='caflow/datasets/ffhq', help='path to images')
    parser.add_argument('--val-batch', type=int, default=20, help='val batch size')
    parser.add_argument('--val-workers', type=int, default=4, help='val_workers')

    #the rest are related to the specific dataset and the required transformations
    parser.add_argument('--create-dataset', default=False, action='store_true')
    parser.add_argument('--dataset-style', type=str, default='inpainting', help='identifier of the stored structure of the dataset')
    parser.add_argument('--max-dataset-size', type=int, default=500, help='Maximum number of samples allowed per dataset. \
                                                                                Set to float("inf") if you want to use the entire training dataset')
    parser.add_argument('--load-size', type=int, default=64)
    parser.add_argument('--mask-to-area', type=float, default=0.15, help='mask to total area ratio in impainting tasks.')
    parser.add_argument('--mask-type', type=str, default='central', help='mask type for inpainting. Supported types: central, half.')
    parser.add_argument('--sr-factor', type=int, default=4, help='sr_factor x super-resolution. Default: 4x.')

    # Testing
    parser.add_argument('--experiment', type=int, default=0, help='which experiment to test.')
    parser.add_argument('--num-samples', type=int, default=8, help='num of samples to generate in testing.')
    parser.add_argument('--num-selected-samples', type=int, default=4, help='num of selected samples based on conditional log-likelihood.')
    parser.add_argument('--running-average', type=int, default=10, help='Average the output image over running_average sampled latents.')
    parser.add_argument('--plot', default=False, action='store_true', help='whether to plot the generated samples.')
    
    # parameteres related to sampling from the annealed latent distribution
    parser.add_argument('--temperatures', type=float, nargs="+", help='List of temperature ranges to anneal the latent distribution.')

    ## parameters related to NUTS algorithim for sampling from the annealed modelled distribution.
    parser.add_argument('--T', type=float, default=0.97, help='Annealing temperature.')
    parser.add_argument('--burn', type=int, default=500, help='Burn-in time in NUTS algorithm.')

    args = parser.parse_args()

    main(args)
    #annealed_distribution_uflow(args)
    
