# We provide  a function that receives as an input the kind of paired MRI, 
# PET dataset that we want to create and creates it.

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

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
            session_date = datetime.datetime(year, month, day)

            immediate_directories_list = os.listdir(os.path.join(subject_dir, session))
            if 'pet' not in immediate_directories_list \
                or 'anat' not in immediate_directories_list:
                continue
            else:
                if os.path.exists(os.path.join(subject_dir, session, 'pet')):
                    pet_scan = '%s_%s_acq-FDG_run-1_%s.nii.gz' % (subject, session, pet_dataset_type)

                    try:
                        assert pet_scan in os.listdir(os.path.join(subject_dir, session, 'pet')), '%s not in the preprocessed files.' % pet_dataset_type
                    except AssertionError:
                        print('%s does not contain the required preprocessed file: %s in %s/pet' % (subject, pet_dataset_type, session))

                    path = os.path.join(subject_dir, session, 'pet', pet_scan)
                    info[subjectID]['pet'].append([session_date, path])
                
                if os.path.exists(os.path.join(subject_dir, session, 'anat')):
                    mri_scan = 'T1_to_%s.nii.gz' % mri_dataset_type

                    try:
                        assert mri_scan in os.listdir(os.path.join(subject_dir, session, 'anat', '%s_%s_acq-T1w_run-1.anat' % (subject, session))), '%s not in the preprocessed files.' % mri_dataset_type
                    except AssertionError:
                        print('%s does not contain the required preprocessed file: %s in %s/anat/' % (subject, mri_dataset_type, session))

                    path = os.path.join(subject_dir, session, 'anat', '%s_%s_acq-T1w_run-1.anat' % (subject, session), mri_scan)
                    info[subjectID]['mri'].append([session_date, path])

    return info

def pairs_for_time_threshold(T : int, info: dict):
    #Inputs: 1.) T: max time difference between the acquisition of the PET and the MRI scans
    #        2.) The information that we have collected by inspecting the dataset (dictionary)

    num_pairs = 0
    for subjectID in info.keys():
        if (not info[subjectID]['pet']) or (not info[subjectID]['mri']):
            continue
        else:
            pet_dates = [x[0] for x in info[subjectID]['pet']]
            mri_dates = [x[0] for x in info[subjectID]['mri']]

            for pet_date in pet_dates:
                for mri_date in mri_dates:
                    delta = pet_date - mri_date
                    print('days difference: %d' % abs(delta.days))
                    if abs(delta.days) <= T:
                        num_pairs+=1
    return num_pairs

def main(args):
    info = inspect_data(args.input_dir, args.output_dir, args.dataset_type)
    print(len(info.keys())

    plt.figure()
    plt.title('Number of pairs as a function of time between acquistion threshold')
    time_thresholds = np.arange(1, args.max_time_threshold)
    num_pairs = [pairs_for_time_threshold(T, info) for T in time_thresholds]
    plt.plot(time_thresholds, num_pairs)
    plt.savefig('num_pairs_function_of_acquistion_time_threshold.png')

    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='/mnt/zfs/Cohort_Raw_Data/ALL_ADNI/T1wPET/t12pet')
    parser.add_argument('--output-dir', type=str, default='/home/gb511/MRI2PET/CAFLOW/caflow/datasets')
    parser.add_argument('--dataset-type', type=str, default='linear-MNI-sliced')

    #inspection settings
    parser.add_argument('--max-time-threshold', type=int, default=40, help='Maximum difference between MRI and PET')

    args = parser.parse_args()
    main(args)