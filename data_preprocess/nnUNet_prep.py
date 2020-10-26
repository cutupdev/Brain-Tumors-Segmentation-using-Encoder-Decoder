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
import os, sys, json, re, boto3
from shutil import copyfile
from collections import OrderedDict

if __name__ == '__main__':
    # use_s3 = True # Use s3 bucket
    # if use_s3:
    #
    # else:


    dataDir = '/home/ubuntu/data/brats-data/nnUNet_raw_data'
    training_data_path = '/home/ubuntu/data/brats-data/MICCAI_BraTS2020_TrainingData'
    validation_data_path = '/home/ubuntu/data/brats-data/MICCAI_BraTS2020_ValidationData'

    # dataDir = '/Users/wslee-2/Data/brats-data/nnUNet_raw_data'
    # training_data_path = '/Users/wslee-2/Data/brats-data/MICCAI_BraTS2020_TrainingData'
    # validation_data_path = '/Users/wslee-2/Data/brats-data/MICCAI_BraTS2020_ValidationData'

    preprocessed_data_path = os.path.join(dataDir,'Task001_BrainTumour')
    preprocessed_data_path_train = os.path.join(preprocessed_data_path, 'imagesTr')
    preprocessed_data_path_train_labels = os.path.join(preprocessed_data_path, 'labelsTr')
    preprocessed_data_path_test = os.path.join(preprocessed_data_path, 'imagesTs')

    # Make directories if they don't exist
    for dir in [preprocessed_data_path,
                 preprocessed_data_path_train,
                 preprocessed_data_path_train_labels,
                 preprocessed_data_path_test]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # Initialize
    numTraining = 0
    numTest = 0
    training = []
    test = []
    corrupted_dirs = []

    # Process training data
    for root, dirs, files in os.walk(training_data_path):
        for name in dirs:
            if 'BraTS20_Training_' in name:
                subject_str = name.split('_')[-1]
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

                for item in filepaths:
                    if not os.path.isfile(filepaths[item]):
                        print(filepaths[item])
                        print('ERRROR: file not found')
                        corrupted_dirs.append(name)
                        # todo: add break to skip subject if corrupted

                # All files in place, copy and add to new directory structure
                # preprocessed_path = os.path.join(preprocessed_data_path_train, )
                #     Check if file has already been copied to new location
                for item in filepaths:
                    if item == 'flair':
                        index = 0
                    elif item == 't1w':
                        index = 1
                    elif item == 't1gd':
                        index = 2
                    elif item == 't2':
                        index = 3
                    elif item == 'label':
                        index = 4
                    else:
                        print('ERROR: {} not valid file name')
                        sys.exit(-1)


                    if index < 4:
                        new_fname = 'BRATS_{}_{:04d}.nii.gz'.format(subject_str, index)
                        new_path = os.path.join(preprocessed_data_path_train,new_fname)
                    else:
                        new_fname = 'BRATS_{}.nii.gz'.format(subject_str)
                        new_path = os.path.join(preprocessed_data_path_train_labels, new_fname)

                    reprocess = False
                    # only copy file if either the file doesn't exist or reprocess is True
                    if reprocess:
                        copyfile(filepaths[item], new_path)
                    else:
                        if not os.path.isfile(new_path):
                            copyfile(filepaths[item], new_path)

                # Generate items for json file
                numTraining += 1
                training.append({"image" : "./imagesTr/BRATS_{}.nii.gz".format(subject_str),
                                 "label" : "./labelsTr/BRATS_{}.nii.gz".format(subject_str)})

    print('training files processed: {}'.format(numTraining))

    ### Process test data (will use validation set)

    # print(training)

    # Create output dictionary to get written to json
    output_dict = OrderedDict([
        ("name", "BRATS"),
        ("description", "Gliomas segmentation tumour and oedema in on brain images"),
        ("reference", "https://www.med.upenn.edu/cbica/brats2020/data.html"),
        ("license", "???"),
        ("release", "???"),
        ("tensorImageSize", "4D"),
        ("modality", {
            "0": "FLAIR",
            "1": "T1w",
            "2": "t1gd",
            "3": "T2w"
        }),
        ("labels", {
            "0": "background",
            "1": "non-enhancing tumor",
            "2": "edema",
            "4": "enhancing tumour"
        }),
        ("numTraining", numTraining),
        ("numTest", numTest),
        ("training", training),
        ("test", [])
    ]
    )

    # Convert to json
    json_out = json.dumps(output_dict)

    # Clean up json file
    json_out = re.sub('}$', '\n}', json_out)
    json_out = re.sub('{', '{\n', json_out, 1)
    json_out = re.sub(r', "', r',\n"', json_out)
    json_out = re.sub(r',\n"label"', r',"label"', json_out)
    json_out = re.sub(r'"modality": {','"modality": {\n',json_out)
    json_out = re.sub(r', "labels": {', ',\n"labels": {\n', json_out)
    json_out = re.sub(r'"0"', '   "0"', json_out)
    json_out = re.sub(r'"1"', '   "1"', json_out)
    json_out = re.sub(r'"2"', '   "2"', json_out)
    json_out = re.sub(r'"3"', '   "3"', json_out)
    json_out = re.sub(r'"4"', '   "4"', json_out)
    json_out = re.sub(r'"T2w"', r'"T2w"\n', json_out)
    json_out = re.sub(r'"enhancing tumour"', r'"enhancing tumour"\n', json_out)
    json_out = re.sub(r'{   "0":', r'{\n   "0":', json_out)






    # json_out.replace('",','",\n')



    # json_out.replace('{','{\n',1)
    # json_out.replace('{', '{\n', 1)
    # print(json_out)

    json_fname = os.path.join(preprocessed_data_path, 'dataset.json')
    with open(json_fname, 'w') as fid:
        fid.write(json_out)









    # print(flair_file)
                # print(os.path.isfile(flair_file))