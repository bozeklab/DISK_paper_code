import os, sys
import tqdm
from glob import glob

import numpy as np
import pandas as pd
from skimage.io import imread, imsave

import matplotlib.pyplot as plt
import matplotlib
if os.uname().nodename == 'france-XPS':
    matplotlib.use('TkAgg')
    basedir = '/home/france/Mounted_dir'
else:
    matplotlib.use('Agg')
    basedir = '/projects/ag-bozek/france'


if __name__ == '__main__':
    name = 'Rat7M'
    name_folder = f'{name}_optipose'
    dataset_dir = os.path.join(basedir, 'results_behavior/datasets', name_folder)
    constant_file_path = os.path.join(dataset_dir, 'constants.py')

    for suffix in ['train', 'val']:
        input_file = '.csv'

        df = pd.read_csv(input_file, sep='|')

        # create the npz file
        input_data = df['input'].values
        gt_data = df['label'].values
        lengths = [input_data.shape[1]] * input_data.shape[0]

        outputfile = os.path.join(dataset_dir, f'{suffix}_dataset_w-0-nans')
        print(f'saving in {outputfile}...')
        np.savez(outputfile, X=input_data, ground_truth=gt_data, lengths=lengths)

        if suffix == 'train':
            # create the constants file, only one per dataset
            keypoints = []
            divider = 3
            freq = 30
            stride = 1
            file_type = 'optipose'
            dlc_likelihood_threshold = 1

            with open(constant_file_path, 'w') as opened_file:
                txt = f"NUM_FEATURES = {input_data.shape[2]}\n"
                txt += f"KEYPOINTS = {keypoints}\n"
                # DIVIDER= 2 for 2D, 3 for 3D, sometimes additional dimension for a confidence score or an error
                # score for the detection
                txt += f"DIVIDER = {divider}\n"
                txt += f"ORIG_FREQ = {freq}\n"
                txt += f"FREQ = {freq}\n"
                txt += f"SEQ_LENGTH = {lengths[0]}\n"
                txt += f"STRIDE = {stride}\n"
                txt += f"W_RESIDUALS = False\n"  # for compatibility reasons (see dataset classes)
                txt += f"FILE_TYPE = '{file_type}'\n"
                txt += f"DLC_LIKELIHOOD_THRESHOLD = {dlc_likelihood_threshold}"
                opened_file.write(txt)
