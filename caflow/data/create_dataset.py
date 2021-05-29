#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 18:12:48 2021

@author: gbatz97
"""


from PIL import Image, ImageOps

from caflow.data.aligned_dataset import AlignedDataset
from caflow.data.image_folder import make_dataset
from caflow.data.image_folder import is_image_file
import os
import numpy as np

def center_crop(img, crop_left, crop_right, crop_top, crop_bottom):
    width, height = img.size
    left = crop_left
    right = width - crop_right
    top = crop_top
    bottom = height - crop_bottom
    return img.crop((left, top, right, bottom))

def create_dataset(master_path='caflow/datasets/edges2shoes', resize_size=32, dataset_size=2000, dataset_style='image2image', mask_to_area=0.1, mask_type='central', sr_factor=4):
    phases_to_create = inspect_dataset(master_path, resize_size, dataset_size) 
    if not phases_to_create:
        print('Datasets already in place.')
    else:
        for phase in phases_to_create:
            print('Create {} dataset'.format(phase))
            data_paths = make_dataset(os.path.join(master_path, phase))
            if dataset_style == 'image2image':
                for i in range(min(dataset_size, len(data_paths))):
                    if (i+1) % 1000 == 0:
                        print(i+1)
                    AB_path = data_paths[i]
                    basename = os.path.basename(AB_path)
                    # read, crop, resize, save
                    
                    # read
                    AB = Image.open(AB_path).convert('RGB')
                    # crop
                    w, h = AB.size
                    w2 = int(w / 2)
                    A = AB.crop((0, 0, w2, h))
                    B = AB.crop((w2, 0, w, h))
                    # resize
                    if isinstance(resize_size, int):
                        resize_size = (resize_size, resize_size)
                    A_resize = A.resize(resize_size, Image.BICUBIC)
                    B_resize = B.resize(resize_size, Image.BICUBIC)
                    # save
                    A_resize.save(os.path.join(master_path, phase, 'A', basename))
                    B_resize.save(os.path.join(master_path, phase, 'B', basename))
                    
            elif dataset_style in ['inpainting', 'Inpainting']:
                dataset = master_path.split('/')[-1]
                print(dataset)
                for i in range(min(dataset_size, len(data_paths))):
                    if (i+1) % 1000 == 0:
                        print(i+1)
                    
                    #------- read -------
                    img_path = data_paths[i]
                    basename = os.path.basename(img_path)
                    img = Image.open(img_path).convert('RGB')

                    ###----new experiment----
                    #centrally crop
                    if dataset == 'ffhq':
                        w, h = img.size
                        img = img.crop((int(0.1*w), int(0.1*h), int(w-0.1*w), int(h-0.1*h))) #-> used for FFHQ
                    elif dataset in ['CelebA', 'celebA']:
                        img = center_crop(img, 40, 40, 60, 30)
                        #print('only for testing paper images')
                    
                    #resize
                    if isinstance(resize_size, int):
                        resize_size = (resize_size, resize_size)

                    img_resized = img.resize(resize_size, Image.BICUBIC)
                    A = img_resized.copy()
                    B = img_resized

                    if mask_type == 'central':
                        #apply the central mask
                        new_w, new_h = img_resized.size
                        mask_len = int(np.sqrt(new_w*new_h*mask_to_area))
                        
                        x1 = new_w//2-mask_len//2
                        x2 = new_h//2-mask_len//2
                        for i in range(mask_len):
                            for j in range(mask_len):
                                A.putpixel((x1+i, x2+j), (0,0,0))
                    
                    elif mask_type == 'half':
                        #apply a vertical mask
                        new_w, new_h = img_resized.size
                        for i in range(new_w//2, new_w):
                            for j in range(new_h):
                                A.putpixel((i, j), (0,0,0))
                        

                    # ------ save ------
                    A.save(os.path.join(master_path, phase, 'A', basename))
                    B.save(os.path.join(master_path, phase, 'B', basename))
            
            elif dataset_style in ['colorisation', 'colorization', 'Colorisation', 'Colorization']:
                print('Creating the colorisation dataset.')
                for i in range(min(dataset_size, len(data_paths))):
                    if (i+1) % 1000 == 0:
                        print(i+1)
                    
                    #------- read -------
                    img_path = data_paths[i]
                    basename = os.path.basename(img_path)
                    print(basename)
                    img = Image.open(img_path).convert('RGB')
                    # ----- resize -----
                    if isinstance(resize_size, int):
                        resize_size = (resize_size, resize_size)
                    img_resized = img.resize(resize_size, Image.BICUBIC)
                    # ---- convert to grayscale ----
                    A = ImageOps.grayscale(img_resized.copy())
                    B = img_resized
                    # ------ save ------
                    if basename.split('.')[-1] is not 'png':
                        basename = basename.split('.')[0]+'.png'

                    A.save(os.path.join(master_path, phase, 'A', basename))
                    B.save(os.path.join(master_path, phase, 'B', basename))
            
            elif dataset_style in ['super-resolution', 'Super-resolution']:
                dataset = master_path.split('/')[-1]
                print('Creating the Super-resolution dataset.')
                for i in range(min(dataset_size, len(data_paths))):
                    if (i+1) % 1000 == 0:
                        print(i+1)
                    
                    #------- read -------
                    img_path = data_paths[i]
                    basename = os.path.basename(img_path)
                    img = Image.open(img_path).convert('RGB')

                    if dataset == 'celebA':
                        #img = center_crop(img, 40, 40, 60, 30)
                        img = center_crop(img, 9, 9, 39, 19) #HR -> (160,160)
                        size2resize = img.size
                        img_resized = img.resize(size2resize, Image.BICUBIC)
                    else:
                        # ----- resize -----
                        if isinstance(resize_size, int):
                            size2resize = (resize_size, resize_size)
                        img_resized = img.resize(size2resize, Image.BICUBIC)

                    # ---- convert to LR ----
                    A = img_resized.copy()
                    A = A.resize((size2resize[0]//sr_factor, size2resize[1]//sr_factor), Image.BICUBIC)
                    A = A.resize(size2resize, Image.NEAREST)
                    B = img_resized

                    # ------ save ------
                    A.save(os.path.join(master_path, phase, 'A', basename))
                    B.save(os.path.join(master_path, phase, 'B', basename))


                    



                    



def inspect_dataset(master_path, resize_size, dataset_size):
    info = {'train':{'A':{'count':0, 'names':[], 'size':None}, \
                     'B':{'count':0, 'names':[], 'size':None}}, \
              'val':{'A':{'count':0, 'names':[], 'size':None}, \
                     'B':{'count':0, 'names':[], 'size':None}}}
    
    for phase in ['train', 'val']:
        for domain in ['A', 'B']:
            subpath=os.path.join(master_path, phase, domain)
            if not os.path.exists(subpath):
                   os.mkdir(subpath)
            else:
                i=0
                for root, _, fnames in os.walk(subpath):
                    phase_domain_names = []
                    for fname in fnames:    
                        if is_image_file(fname):
                            info[phase][domain]['count'] += 1
                            phase_domain_names.append(os.path.basename(fname))
                            if i==0:
                                img = Image.open(os.path.join(subpath, fname)).convert('RGB')
                                w, h = img.size
                                info[phase][domain]['size'] = w #we assume w = h
                            i+=1
                    
                    info[phase][domain]['names'] = sorted(phase_domain_names)

    empty = {'train':True, 'val':True}
    for phase in ['train', 'val']:
        if info[phase]['A']['count']>0 or info[phase]['B']['count']>0:
            empty[phase]=False

    for phase in ['train', 'val']:
        try:
            #check the count
            assert info[phase]['A']['count'] == info[phase]['B']['count'], \
            'Different count number between A and B domains.'

            if phase == 'train':
                assert info[phase]['A']['count'] == dataset_size, 'Dataset size different than requested.'

            #check image size
            assert info[phase]['A']['size'] == resize_size, 'Domain A has different size than size requested'
            assert info[phase]['B']['size'] == resize_size, 'Domain B has different size than size requested'

            #check image pairing
            for i in range(info[phase]['A']['count']):
                assert info[phase]['A']['names'][i]==info[phase]['B']['names'][i], \
                'Wrong image pairing. A:{} - B:{}'.format(info[phase]['A']['names'][i], info[phase]['B']['names'][i])

        except AssertionError:
            for domain in ['A', 'B']:
                subpath = os.path.join(master_path, phase, domain)
                for root, _, fnames in os.walk(subpath):
                    for fname in fnames:
                        os.remove(os.path.join(subpath, fname))
                
            empty[phase]=True
        
    datasets_to_create = []
    for phase in ['train', 'val']:
        if empty[phase] == True:
            datasets_to_create.append(phase)
        
    return datasets_to_create

def load_image_paths(master_path, phase):
    assert os.path.isdir(os.path.join(master_path, phase)), '%s is not a valid directory' % dir
    #print('load img dir: {}'.format(os.path.join(master_path, phase)))
    domains = ['A', 'B']
    images = {}
    for domain in domains:
        for root, _, fnames in os.walk(os.path.join(master_path, phase, domain)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    if os.path.basename(path) not in images.keys():
                        images[os.path.basename(path)]=[]
                        images[os.path.basename(path)].append(path)
                    else:
                        images[os.path.basename(path)].append(path)
    
    #for key in list(images.keys())[:10]:
    #    print('{} - {} - {}'.format(key, images[key][0], images[key][1]))

    load_images = {'A':[], 'B':[]}
    for key in images.keys():
        load_images['A'].append(images[key][0])
        load_images['B'].append(images[key][1])

    print(load_images['A'][:3])
    print(load_images['B'][:3])

    #assertions
    assert len(load_images['A'])==len(load_images['B']), 'There is a mismatch in the number of domain A and domain B images.'
    for i in range(len(load_images['A'])):
        assert os.path.basename(load_images['A'][i])==os.path.basename(load_images['B'][i]), \
               'The images are not paired. A:{} - B:{}'.format(os.path.basename(load_images['A'][i]), os.path.basename(load_images['B'][i]))

    return load_images