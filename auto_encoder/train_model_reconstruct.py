#!/usr/bin/env python
#
# train_model.py
#
# Created by Scott Lee on 10/27/20.
# Adapted from Colab: https://colab.research.google.com/github/IAmSuyogJadhav/3d-mri-brain-tumor-segmentation-using-autoencoder-regularization/blob/master/Example_on_BRATS2018.ipynb
#

'''
Description: train model built in model.py
'''

import SimpleITK as sitk  # For loading the dataset
import numpy as np  # For data manipulation
from model_reconstruct import build_model  # For creating the model
import glob  # For populating the list of files
from scipy.ndimage import zoom  # For resizing
import re  # For parsing the filenames (to know their modality)
import os, pickle
from keras.callbacks import History, ModelCheckpoint, CSVLogger
from datetime import datetime
from keras.models import load_model

def read_img(img_path):
    """
    Reads a .nii.gz image and returns as a numpy array.
    """
    return sitk.GetArrayFromImage(sitk.ReadImage(img_path))


def resize(img, shape, mode='constant', orig_shape=(155, 240, 240)):
    """
    Wrapper for scipy.ndimage.zoom suited for MRI images.
    """
    assert len(shape) == 3, "Can not have more than 3 dimensions"
    factors = (
        shape[0] / orig_shape[0],
        shape[1] / orig_shape[1],
        shape[2] / orig_shape[2]
    )

    # Resize to the given shape
    return zoom(img, factors, mode=mode)


def preprocess(img, out_shape=None):
    """
    Preprocess the image.
    Just an example, you can add more preprocessing steps if you wish to.
    """
    if out_shape is not None:
        img = resize(img, out_shape, mode='constant')

    # Normalize the image
    mean = img.mean()
    std = img.std()
    return (img - mean) / std


def preprocess_label(img, out_shape=None, mode='nearest'):
    """
    Separates out the 3 labels from the segmentation provided, namely:
    GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2))
    and the necrotic and non-enhancing tumor core (NCR/NET — label 1)
    """
    ncr = img == 1  # Necrotic and Non-Enhancing Tumor (NCR/NET)
    ed = img == 2  # Peritumoral Edema (ED)
    et = img == 4  # GD-enhancing Tumor (ET)

    if out_shape is not None:
        ncr = resize(ncr, out_shape, mode=mode)
        ed = resize(ed, out_shape, mode=mode)
        et = resize(et, out_shape, mode=mode)

    return np.array([ncr, ed, et], dtype=np.uint8)


if __name__ == '__main__':
    DEBUG = True # Load only a few images (4) and train for a few epochs (3)
    REDUCE_MODALITIES = True  # Select this as True to drop low priority modalities

    # timestamp experiment to organize results
    timestamp = datetime.today().strftime('%Y-%m-%d-%H%M')
    timestamp = str(timestamp)

    output_path_dict = {'aws': '/home/ubuntu/',
                        'colab': '/gdrive/Shared drives/CS230 - Term Project/results'}


    output_path = output_path_dict['colab']
    output_path = os.path.join(output_path,'reconstruct','experiment_{}'.format(timestamp))
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Re-load existing model from Keras checkpoint and resume training. If reload_path = '', train as new model
    # reload_path = '/home/ubuntu/checkpoints/model_training_11_03/ae_weights.400-0.00843.hdf5'
    reload_path = ''

    # path = '/gdrive/Shared drives/CS230 - Term Project/data/BraTS_2018/MICCAI_BraTS_2018_Data_Training/'
    # path = '/Users/wslee-2/Data/brats-data/MICCAI_BraTS_2018_Data_Training'

    #################### 2018 data #######################
    path = '/gdrive/Shared drives/CS230 - Term Project/data/BraTS_2018/MICCAI_BraTS_2018_Data_Training/'
    # path = '/home/ubuntu/data/brats-data/MICCAI_BraTS_2018_Data_Training'
    #
    # Import data
    # Get a list of files for all modalities individually
    t1 = glob.glob(os.path.join(path, '*GG/*/*t1.nii.gz'))
    t2 = glob.glob(os.path.join(path, '*GG/*/*t2.nii.gz'))
    flair = glob.glob(os.path.join(path, '*GG/*/*flair.nii.gz'))
    t1ce = glob.glob(os.path.join(path, '*GG/*/*t1ce.nii.gz'))
    seg = glob.glob(os.path.join(path, '*GG/*/*seg.nii.gz'))  # Ground Truth

    pat = re.compile('.*_(\w*)\.nii\.gz')

    ###################### 2020 data ####################

    # path = '/home/ubuntu/data/brats-data/MICCAI_BraTS2020_TrainingData'
    #
    # # Import data
    # # Get a list of files for all modalities individually
    # t1 = glob.glob(os.path.join(path, '*/*t1.nii.gz'))
    # t2 = glob.glob(os.path.join(path, '*/*t2.nii.gz'))
    # flair = glob.glob(os.path.join(path, '*/*flair.nii.gz'))
    # t1ce = glob.glob(os.path.join(path, '*/*t1ce.nii.gz'))
    # seg = glob.glob(os.path.join(path, '*/*seg.nii.gz'))  # Ground Truth
    #
    # pat = re.compile('.*_(\w*)\.nii\.gz')

    history = History()  # Initialize history to record training loss

    data_paths = [{
        pat.findall(item)[0]: item
        for item in items
    }
        for items in list(zip(t1, t2, t1ce, flair, seg))]

    # Initialize memory
    input_shape = (4, 80, 96, 64)
    #if DEBUG:
    #    input_shape = (4, 80, 96, 64)

    output_channels = 4

    # labels will have the same dimensions as input
    data = np.empty((len(data_paths),) + input_shape, dtype=np.float32)
    labels = np.empty((len(data_paths),) + input_shape, dtype=np.float32)

    if DEBUG:
        data = np.empty((len(data_paths[:4]),) + input_shape, dtype=np.float32)
        labels = np.empty((len(data_paths[:4]),) + input_shape, dtype=np.float32)

    import math
    print('reading images...')
    # Parameters for the progress bar
    total = len(data_paths)

    if DEBUG:
        total = len(data_paths[:4])

    step = 25 / total
    endpoint = -1

    if DEBUG:
        endpoint = 4

    bad_frames = [] # keep list of any frames with nan or inf

    for i, imgs in enumerate(data_paths[:endpoint]):
        try:
            temp = np.array([preprocess(read_img(imgs[m]), input_shape[1:]) for m in ['t1', 't2', 't1ce', 'flair']],
                            dtype=np.float32)
            labels[i] = temp

            # initialize missing modes with zero mean small values
            temp[0, :, :, :] = np.random.normal(loc=0, scale=1, size=temp.shape[0:])
            temp[1, :, :, :] = np.random.normal(loc=0, scale=1, size=temp.shape[1:])

            data[i] = temp

            if ~np.isfinite(data[i]).any() or ~np.isfinite(labels[i]).any():
                print('bad frame found:')
                print(data_paths[i])
                bad_frames.append(i)

            # Print the progress bar
            print('\r' + f'Progress: '
                         f"[{'=' * int((i + 1) * step) + ' ' * (24 - int((i + 1) * step))}]"
                         f"({math.ceil((i + 1) * 100 / (total))} %)",
                  end='')
        except Exception as e:
            print(f'Something went wrong with {imgs["t1"]}, skipping...\n Exception:\n{str(e)}')
            continue

    # Remove any bad frames
    if len(bad_frames) > 0:
        print('removing bad frames:')
        print(bad_frames)
        print('before data.shape: {}'.format(data.shape))
        print('before labels.shape: {}'.format(labels.shape))
        data = np.delete(data, bad_frames, axis=0)
        labels = np.delete(labels, bad_frames, axis=0)
        print('after data.shape: {}'.format(data.shape))
        print('after labels.shape: {}'.format(labels.shape))

    # Model training
    batch_size = 1
    if DEBUG:
        epochs = 3
    else:
        epochs = 400

    # Setup callbacks
    # checkpoint_filepath = '/home/ubuntu/checkpoints/model_ae_{}'.format(timestamp)'
    timestamp = datetime.today().strftime('%Y-%m-%d-%H%M')
    timestamp = str(timestamp)

    # filepath_checkpoint = os.path.join(output_path,'checkpoints','checkpoints/ae_weights.{epoch:03d}-{loss:.5f}.hdf5')
    # filepath_csv = os.path.join(output_path,'checkpoints','log_{}.csv'.format(timestamp))
    filepath_checkpoint = os.path.join(output_path,'ae_weights.{epoch:03d}-{loss:.5f}.hdf5')
    filepath_csv = os.path.join(output_path, 'log_{}.csv'.format(timestamp))

    checkpoint = ModelCheckpoint(filepath = filepath_checkpoint, monitor='loss', verbose=1, save_best_only=True, mode='min')

    csv_logger = CSVLogger(filepath_csv, append=True, separator=',')

    callbacks_list = [checkpoint, csv_logger]

    timestamp = datetime.today().strftime('%Y-%m-%d-%H%M')

    model = build_model(input_shape=input_shape, output_channels=4)

    if reload_path:
        model.load_weights(reload_path)

    model.fit(data, [labels], batch_size=batch_size, epochs=epochs, callbacks=callbacks_list)

    # filepath_model = os.path.join(output_path,'model_ae_{}_{}_tf'.format(epochs, timestamp))
    # filepath_results_dict = os.path.join(output_path,'model_ae_{}_{}_dict'.format(epochs, timestamp))
    filepath_model = os.path.join(output_path, 'model_ae_{}_{}_tf'.format(epochs, timestamp))
    filepath_results_dict = os.path.join(output_path, 'model_ae_{}_{}_dict'.format(epochs, timestamp))
    model.save(filepath_model,save_format='tf')
    print(history.history)
    with open(filepath_results_dict.format(epochs, timestamp), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
