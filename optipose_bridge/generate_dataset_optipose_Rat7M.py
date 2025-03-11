import os
from scipy.io import loadmat
from glob import glob
import pandas as pd
import numpy as np
from cvkit.pose_estimation.data_readers import FlattenedDataStore

from cvkit.pose_estimation.config import PoseEstimationConfig
from OptiPose.model.dataset.generate_dataset import generate_dataset

import matplotlib
if os.uname().nodename == 'france-XPS':
    matplotlib.use('TkAgg')
    basedir = '/home/france/Mounted_dir'
else:
    matplotlib.use('Agg')
    basedir = '/projects/ag-bozek/france'


if __name__ == '__main__':
    is_val = True
    min_sample_length = 30
    name = 'Rat7M'
    name_folder = f'{name}_optipose'
    keypoints_to_drop = ['ElbowL', 'ArmL', 'ElbowR', 'ArmR']


    config = PoseEstimationConfig(f'./example_configs/{name}.yml')
    base_folder = os.path.join(basedir, 'behavior_data/mouse_dannce_dataset')
    output_folder = os.path.join(basedir, 'results_behavior/datasets', name_folder)

    if not is_val:
       mat_files = ['mocap-s2-d1.mat', 'mocap-s2-d2.mat', 'mocap-s3-d1.mat', 'mocap-s5-d1.mat', 'mocap-s5-d2.mat']
       suffix = 'train'
       n_samples = 976_000
    else:
       mat_files = ['mocap-s4-d1.mat', ]
       suffix = 'val'
       n_samples = 60_000

    csv_files = []
    for mf in mat_files:
        mat = loadmat(mf)
        data = np.moveaxis(np.array(list(mat['mocap'][0][0])), 1, 0)

        keypoints = list(mat['mocap'][0][0].dtype.fields.keys())
        indices_to_drop = [keypoints.index(k) for k in keypoints_to_drop]
        indices_to_keep = [i for i in np.arange(len(keypoints)) if i not in indices_to_drop]
        new_data = data[:, indices_to_keep, :]

        ## format into the csv for optipose
        columns = []
        for k in range(len(keypoints)):
            columns.extend([f'{k}_1', f'{k}_2', f'{k}_3'])

        df = pd.DataFrame(columns=columns, data=new_data.reshape(new_data.shape[0], -1))
        csv_file = os.path.join(base_folder, f'{os.path.basename(mf).split(".")[0]}_dropkp.csv')
        df.to_csv(csv_file, index=False)
        csv_files.append(csv_file)


    data_stores = [FlattenedDataStore(config.body_parts, path) for path in csv_files]
    generate_dataset(output_folder, name, is_val, data_stores, n_samples, min_seq_length=min_sample_length,
                 max_seq_length=min_sample_length, min_x=-700,
                 max_x=700, min_y=-700, max_y=700, prefix="", suffix=suffix, random_noise=50)