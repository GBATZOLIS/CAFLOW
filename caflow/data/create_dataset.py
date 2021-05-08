#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 18:12:48 2021

@author: gbatz97
"""


from PIL import Image

from caflow.data.aligned_dataset import AlignedDataset
from caflow.data.image_folder import make_dataset
from caflow.data.image_folder import is_image_file
import os
import numpy as np

def create_dataset(master_path='caflow/datasets/edges2shoes', resize_size=32, dataset_size=2000, dataset_style='BicycleGAN', mask_to_area=0.1):
    phases_to_create = inspect_dataset(master_path, resize_size, dataset_size) 
    if not phases_to_create:
        print('Datasets already in place.')
    else:
        for phase in phases_to_create:
            print('Create {} dataset'.format(phase))
            data_paths = make_dataset(os.path.join(master_path, phase))
            if dataset_style == 'BicycleGAN':
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
            else:
                for i in range(min(dataset_size, len(data_paths))):
                    if (i+1) % 1000 == 0:
                        print(i+1)
                    
                    #------- read -------
                    img_path = data_paths[i]
                    basename = os.path.basename(img_path)
                    img = Image.open(img_path).convert('RGB')
                    # ----- resize -----
                    if isinstance(resize_size, int):
                        resize_size = (resize_size, resize_size)
                    img_resized = img.resize(resize_size, Image.BICUBIC)
                    A = img_resized.copy()
                    B = img_resized
                    #---- apply the mask ---
                    mask_len = int(np.sqrt(resize_size[0]*resize_size[1]*mask_to_area))
                    window1 = int(0.1*resize_size[0])
                    window2 = int(0.1*resize_size[1])
                    x1 = np.random.randint(window1, resize_size[0]-mask_len-window1)
                    x2 = np.random.randint(window2, resize_size[1]-mask_len-window2)

                    for i in range(mask_len):
                        for j in range(mask_len):
                            A.putpixel((x1+i, x2+j), (0,0,0))


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