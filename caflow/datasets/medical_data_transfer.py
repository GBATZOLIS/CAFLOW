# We provide  a function that receives as an input the kind of paired MRI, 
# PET dataset that we want to create and creates it.

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import nibabel as nib
import pickle
from tqdm import tqdm
from scipy.ndimage import zoom
from pathlib import Path
#input_dir = /mnt/zfs/Cohort_Raw_Data/ALL_ADNI/T1wPET/t12pet
#output_dir = /home/gb511/MRI2PET/CAFLOW/caflow/datasets
def inspect_data(input_dir, output_dir, dataset_type):
    #We can test different preprocessed paired forms of the initial dataset.
    # Dataset_types: 
    #                1.) Sliced Linear MNI -> lin_to_MNI_sliced
    #                2.) Sliced Non-Linear MNI -> nonlin_to_MNI_sliced
    #                3.) in Tw1 space -> to_T1w
    #                4.) ROI

    if dataset_type == 'linear-MNI-sliced':
        pet_dataset_type = 'lin_to_MNI_sliced'
        mri_dataset_type = 'MNI_lin_sliced'
    elif dataset_type == 'nonlinear-MNI-sliced':
        pet_dataset_type = 'nonlin_to_MNI_sliced'
        mri_dataset_type = 'MNI_nonlin_sliced'

    info = {}
    for subject in os.listdir(input_dir):
        subjectID = int(subject.split('-')[1]) #subject example: sub-4842
        subject_dir = os.path.join(input_dir, subject)

        info[subjectID] = {'pet': [], 'mri': []}

        for session in os.listdir(subject_dir):
            #session example: ses-2018-05-14
            year, month, day = int(session.split('-')[1]), int(session.split('-')[2]), int(session.split('-')[3])
            session_date = datetime.date(year=year, month=month, day=day)

            if os.path.exists(os.path.join(subject_dir, session, 'pet')):
                pet_scan = '%s_%s_acq-FDG_run-1_%s.nii.gz' % (subject, session, pet_dataset_type)

                try:
                    assert pet_scan in os.listdir(os.path.join(subject_dir, session, 'pet')), '%s not in the preprocessed files.' % pet_dataset_type
                    path = os.path.join(subject_dir, session, 'pet', pet_scan)
                    #name = os.path.basename(path).split('.')[0]
                    info[subjectID]['pet'].append([session_date, path])
                except AssertionError:
                    print('%s does not contain the required preprocessed file: %s in %s/pet' % (subject, pet_dataset_type, session))

            if os.path.exists(os.path.join(subject_dir, session, 'anat')):
                mri_scan = 'T1_to_%s.nii.gz' % mri_dataset_type

                try:
                    assert mri_scan in os.listdir(os.path.join(subject_dir, session, 'anat', '%s_%s_acq-T1w_run-1.anat' % (subject, session))), '%s not in the preprocessed files.' % mri_dataset_type
                    path = os.path.join(subject_dir, session, 'anat', '%s_%s_acq-T1w_run-1.anat' % (subject, session), mri_scan)
                    #name = os.path.basename(path).split('.')[0]
                    info[subjectID]['mri'].append([session_date, path])
                except AssertionError:
                    print('%s does not contain the required preprocessed file: %s in %s/anat/' % (subject, mri_dataset_type, session))

    return info

def pairs_for_time_threshold(T : int, info: dict):
    #Inputs: 1.) T: max time difference between the acquisition of the PET and the MRI scans
    #        2.) The information that we have collected by inspecting the dataset (dictionary)

    num_pairs = 0
    paths_of_accepted_pairs = []
    renamed_paired_scans = []
    for subjectID in info.keys():
        if (not info[subjectID]['pet']) or (not info[subjectID]['mri']):
            continue
        else:
            pet_dates = [x[0] for x in info[subjectID]['pet']]
            pet_paths = [x[1] for x in info[subjectID]['pet']]
            mri_dates = [x[0] for x in info[subjectID]['mri']]
            mri_paths = [x[1] for x in info[subjectID]['mri']]

            for pet_date, pet_path in zip(pet_dates, pet_paths):
                for mri_date, mri_path in zip(mri_dates,mri_paths):
                    delta = pet_date - mri_date
                    #print(mri_date)#print(pet_date)#print('Subject %d -> days difference: %d' % (subjectID, abs(delta.days)))
                    if abs(delta.days) <= T:
                        num_pairs+=1
                        paths_of_accepted_pairs.append([mri_path, pet_path])
                        renamed_paired_scans.append(['%d.npy' % num_pairs, '%d.npy' % num_pairs])

    return num_pairs, paths_of_accepted_pairs, renamed_paired_scans

def inspect_scan(scan, name):
    # inspect an MRI/PET scan -> 1.) Discrete values? Continuous? If discrete, what range?, what discretisation?
    #                            2.) Dimensionality

    #x is a numpy array
    print('Scan shape: ', scan.shape)
    flattened_scan = scan.flatten()
    for x in flattened_scan:
        if isinstance(x, int):
            continue
        else:
            print('Scan contains non-integer values e.g. %.9f' % x)
            break

    unique, counts = np.unique(flattened_scan, return_counts=True)
    print(unique)
    plt.figure()
    plt.title('Count of unique values')
    plt.plot(unique, counts)
    plt.savefig('Frequency_of_unique_values_%s.png' % name)

def read_scan(path):
    scan = nib.load(path)
    scan = scan.get_fdata()
    return scan

def plot_num_pairs_vs_acquisition_threshold(max_time_threshold, info):
    plt.figure()
    plt.title('Number of pairs as a function of time between acquistion threshold')
    time_thresholds = np.arange(1, max_time_threshold)
    num_pairs = []
    for T in time_thresholds:
        number_of_pairs, _, _ = pairs_for_time_threshold(T, info)
        num_pairs.append(number_of_pairs)
    plt.plot(time_thresholds, num_pairs)
    plt.savefig('num_pairs_function_of_acquistion_time_threshold.png')

def save(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def inspect_data_values(paths_of_accepted_pairs):
    unique_vals = {'mri':[], 'pet':[]}
    num_unique_vals = {'mri':[], 'pet':[]}
    scan_unique_vals = {}
    mri_scans, pet_scans = [], []
    for i, paired_path in tqdm(enumerate(paths_of_accepted_pairs)):
        if i>50:
            break

        mri_path, pet_path = paired_path[0], paired_path[1]
        mri_scan, pet_scan = read_scan(mri_path), read_scan(pet_path)
        mri_scans.append(mri_scan)
        pet_scans.append(pet_scan)

        scan_unique_vals['mri'], scan_unique_vals['pet'] = np.unique(mri_scan), np.unique(pet_scan)
        for modality in unique_vals.keys():
            extended_unique_vals = unique_vals[modality].copy()
            extended_unique_vals.extend(scan_unique_vals[modality])
            unique_extended = list(set(extended_unique_vals))
            unique_vals[modality] = unique_extended
            num_unique_vals[modality].append(len(unique_vals[modality]))
    
    mri_scans, pet_scans = np.stack(mri_scans), np.stack(pet_scans)
    
    #plotting
    print('Plotting unique values vs number of scans')
    for modality in unique_vals.keys():
        plt.figure()
        plt.title('%s unique values vs number of scans' % modality)
        plt.plot(np.arange(1,len(num_unique_vals[modality])+1), num_unique_vals[modality])
        plt.savefig('%s_uniquevaluesvsnumberofscans.png' % modality)
    
    mri_unique, mri_count = np.unique(mri_scans, return_counts=True)
    pet_unique, pet_count = np.unique(pet_scans, return_counts=True)

    print('Plotting count of unique values for both modalities')
    #mri
    plt.figure()
    plt.title('Count of unique values (MRI)')
    plt.plot(mri_unique, mri_count)
    plt.savefig('Counts_of_unique_values_MRI.png')
    print('---MRI---')
    print('---Minimum value: %.12f - count: %d' % (mri_unique[0], mri_count[0]))
    print('---Just above Minimum value: %.12f - count: %d' % (mri_unique[1], mri_count[1]))
    print('---Maximum value: %.12f - count: %d' % (mri_unique[-1], mri_count[-1]))
    #pet
    plt.figure()
    plt.title('Count of unique values (PET)')
    plt.plot(pet_unique, pet_count)
    plt.savefig('Counts_of_unique_values_PET.png')
    print('---PET---')
    print('---Minimum value: %.12f - count: %d' % (pet_unique[0], pet_count[0]))
    print('---Just above Minimum value: %.12f - count: %d' % (pet_unique[1], pet_count[1]))
    print('---Maximum value: %.12f - count: %d' % (pet_unique[-1], pet_count[-1]))


'''2'''
# Create a function that receives as input the paired paths 
# and copies them to the correct directory for training, validation and testing

def prepare_training_dataset(output_dir, read_paths, save_names, target_resolution=(96,96,96), split=[0.8, 0.1, 0.1]):
    def calculate_zoom(target_shape, original_shape):
        zoom = []
        for x in range(len(original_shape)):
            if target_resolution[x] == -1:
                zoom.append(1)
            else:
                zoom.append(target_resolution[x]/mri_scan_shape[x])
        return zoom

    def save_mri_pet_paired_scans(phase, mri_scan, pet_scan, mri_scan_name, pet_scan_name):
        mri_save_path = os.path.join(output_dir, 'mri2pet', phase, 'A', mri_scan_name)
        pet_save_path = os.path.join(output_dir, 'mri2pet', phase, 'B', pet_scan_name)
        np.save(mri_save_path, mri_scan)
        np.save(pet_save_path, pet_scan)

    Path(os.path.join(output_dir, 'mri2pet', 'train', 'A')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, 'mri2pet', 'train', 'B')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, 'mri2pet', 'val', 'A')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, 'mri2pet', 'val', 'B')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, 'mri2pet', 'test', 'A')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, 'mri2pet', 'test', 'B')).mkdir(parents=True, exist_ok=True)

    num_pairs = len(read_paths)
    permuted_indices = np.random.permutation(num_pairs)

    mri_min, mri_max = float('inf'), float('-inf')
    pet_min, pet_max = float('inf'), float('-inf')
    for i, index in tqdm(enumerate(permuted_indices)):
        mri_path, pet_path = read_paths[index][0], read_paths[index][1]
        mri_scan, pet_scan = read_scan(mri_path), read_scan(pet_path)
        mri_scan_shape, pet_scan_shape = mri_scan.shape, pet_scan.shape
        
        resized_mri_scan = zoom(mri_scan, zoom = calculate_zoom(target_resolution, mri_scan_shape))
        print('mri shape: ', resized_mri_scan.shape)

        scan_min, scan_max = np.min(resized_mri_scan), np.max(resized_mri_scan)
        if scan_min < mri_min:
            mri_min = scan_min
        if scan_max > mri_max:
            mri_max = scan_max

        resized_pet_scan = zoom(pet_scan, zoom = calculate_zoom(target_resolution, pet_scan_shape))
        print('pet shape: ', resized_pet_scan.shape)

        scan_min, scan_max = np.min(resized_pet_scan), np.max(resized_pet_scan)
        if scan_min < pet_min:
            pet_min = scan_min
        if scan_max > pet_max:
            pet_max = scan_max

        #save the scan in the right folder.
        '''
        if i < int(split[0]*num_pairs):
            save_mri_pet_paired_scans('train', resized_mri_scan, resized_pet_scan, save_names[index][0], save_names[index][1])#save under train
        elif i >= int(split[0]*num_pairs) and i < int((split[0]+split[1])*num_pairs):
            save_mri_pet_paired_scans('val', resized_mri_scan, resized_pet_scan, save_names[index][0], save_names[index][1])#save under val
        else:
            save_mri_pet_paired_scans('test', resized_mri_scan, resized_pet_scan, save_names[index][0], save_names[index][1])#save under test
        '''

    print('MRI RANGE: (%.8f, %.8f)' % (mri_min, mri_max))
    print('PET RANGE: (%.8f, %.8f)' % (pet_min, pet_max))

def main(args):
    if args.load_info:
        info = load('dataset_info')
    else:
        info = inspect_data(args.input_dir, args.output_dir, args.dataset_type)
        save(info, 'dataset_info')

    plot_num_pairs_vs_acquisition_threshold(args.inspect_time_threshold, info)
    
    num_pairs, paths_of_accepted_pairs, renamed_paired_scans = pairs_for_time_threshold(args.time_threshold, info)
    prepare_training_dataset(args.output_dir, paths_of_accepted_pairs, renamed_paired_scans, target_resolution=args.target_resolution, split=args.split)

    print('============ dataset information ==============')
    print('----- training examples: %d' % int(args.split[0]*num_pairs))
    print('----- validation examples: %d' % int(args.split[1]*num_pairs))
    print('----- testing examples: %d' % int(args.split[2]*num_pairs))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='/mnt/zfs/Cohort_Raw_Data/ALL_ADNI/T1wPET/t12pet')
    parser.add_argument('--output-dir', type=str, default='/home/gb511/MRI2PET/CAFLOW/caflow/datasets')
    parser.add_argument('--dataset-type', type=str, default='linear-MNI-sliced')

    #inspection settings
    parser.add_argument('--load-info', default=False, action='store_true', help='Load inspection info. If not loaded, it will be generated. Default=False')
    parser.add_argument('--inspect-time-threshold', type=int, default=90, help='Maximum difference between MRI and PET')

    #dataset creation settings
    parser.add_argument('--target-resolution', nargs='+', type=int, default=[96,96,96])
    parser.add_argument('--time-threshold', default=35, type=int, help='Maximum time difference between acquisition of MRI and PET scans.')
    parser.add_argument('--split', nargs='+', type=float, default=[0.85, 0.1, 0.05], help='train-val-test split.')

    args = parser.parse_args()
    main(args)