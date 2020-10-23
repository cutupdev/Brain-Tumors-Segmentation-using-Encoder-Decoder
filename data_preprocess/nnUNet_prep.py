#!/usr/bin/env python
#
# nnUNet_prep.py
#
# Created by Scott Lee on 10/22/20.
#

'''
Description:
Convert data from downloaded version to version ready for nnUNet processing.
'''
import os, sys

if __name__ == '__main__':
    dataDir = '/Users/wslee-2/Data/brats-data/'
    training_data_path = os.path.join(dataDir,'MICCAI_BraTS2020_TrainingData')
    validation_data_path = os.path.join(dataDir, 'MICCAI_BraTS2020_TrainingData')

    preprocessed_data_path = os.path.join(dataDir,'Task001_BrainTumour')

    # Initialize
    numTraining = 0
    numTest = 0
    training = []
    test = []


    for root, dirs, files in os.walk(training_data_path):
        for name in dirs:
            if 'BraTS20_Training_' in name:
                subject_dir = os.path.join(root, name)
                print(subject_dir)
                print(name)
                # Filepaths
                filepaths = {
                    'flair' : os.path.join(subject_dir, name + '_flair.nii.gz'),
                    't1w'   : os.path.join(subject_dir, name + '_t1.nii.gz'),
                    't1gd'  : os.path.join(subject_dir, name + '_t1ce.nii.gz'),
                    't2'    : os.path.join(subject_dir, name + '_t2.nii.gz'),
                    'label' : os.path.join(subject_dir, name + '_seg.nii.gz')
                }
                # flair = os.path.join(subject_dir,name + '_flair.nii.gz')
                # t1w = os.path.join(subject_dir,name + '_t1.nii.gz')
                # t1gd = os.path.join(subject_dir,name + '_t1ce.nii.gz')
                # t2 = os.path.join(subject_dir, name + '_t2.nii.gz')
                # label = os.path.join(subject_dir, name + '_seg.nii.gz')
                print(filepaths)

                for item in filepaths:
                    if not os.path.isfile(filepaths[item]):
                        print(filepaths[item])
                        print('ERRROR: file not found')
                        sys.exit(-1)

                # print(flair_file)
                # print(os.path.isfile(flair_file))