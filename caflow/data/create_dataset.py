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

def create_dataset(master_path='caflow/datasets/edges2shoes', resize_size=32, dataset_size=2000):
    phases_to_create = inspect_dataset(master_path, resize_size, dataset_size) 
    if not phases_to_create:
        print('Datasets already in place.')
    else:
        for phase in phases_to_create:
            print('phase: %s' % phase)
            data_paths = make_dataset(os.path.join(master_path, phase))
            for i in range(min(dataset_size, len(data_paths))):
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
                    for fname in fnames:    
                        if is_image_file(fname):
                            info[phase][domain]['count'] += 1
                            info[phase][domain]['names'].append(os.path.basename(fname))
                            if i==0:
                                img = Image.open(os.path.join(subpath, fname)).convert('RGB')
                                w, h = img.size
                                info[phase][domain]['size'] = w #we assume w = h
                            i+=1

    empty = {'train':True, 'val':True}
    for phase in ['train', 'val']:
        if info[phase]['A']['count']>0 or info[phase]['B']['count']>0:
            empty[phase]=False

    for phase in ['train', 'val']:
        try:
            #check the count
            assert info[phase]['A']['count'] == info[phase]['B']['count']
            if phase == 'train':
                assert info[phase]['A']['count'] == dataset_size

            #check image size
            assert info[phase]['A']['size'] == resize_size
            assert info[phase]['B']['size'] == resize_size

            #check image pairing
            for i in range(info[phase]['A']['count']):
                assert info[phase]['A']['names'][i]==info[phase]['B']['names'][i]

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
    images={'A':[], 'B':[]}
    for domain in images.keys():
        for root, _, fnames in sorted(os.walk(os.path.join(master_path, phase, domain))):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images[domain].append(path)
    #assertions
    assert len(images['A'])==len(images['B']), 'There is a mismatch in the number of domain A and domain B images.'
    for i in range(len(images['A'])):
        assert os.path.basename(images['A'][i])==os.path.basename(images['B'][i]), \
               'The images are not paired. A:{} - B:{}'.format(os.path.basename(images['A'][i]), os.path.basename(images['B'][i]))

    return images