import os
from scipy.io import loadmat
from glob import glob
import pandas as pd
import numpy as np
from cvkit.pose_estimation.data_readers import FlattenedDataStore

from cvkit.pose_estimation.config import PoseEstimationConfig
from OptiPose.model.dataset.generate_dataset import generate_dataset
from create_csv_for_optipose import find_holes

import matplotlib
if os.uname().nodename == 'france-XPS':
    matplotlib.use('TkAgg')
    basedir = '/home/france/Mounted_dir'
elif 'ada' in os.uname().nodename:
    matplotlib.use('Agg')
    basedir = '/data/frose1/'
else:
    matplotlib.use('Agg')
    basedir = '/projects/ag-bozek/france'


if __name__ == '__main__':
    min_sample_length = 30
    name = 'Rat7M'
    name_folder = f'{name}_optipose'
    keypoints_to_drop = ['ElbowL', 'ArmL', 'ElbowR', 'ArmR']

    for is_val in [True, False]:

        config = PoseEstimationConfig(f'./example_configs/{name}.yml')
        base_folder = os.path.join(basedir, 'behavior_data/mouse_dannce_dataset')
        output_folder = os.path.join(basedir, 'results_behavior/datasets', name_folder)

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

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
            mat = loadmat(os.path.join(base_folder, mf))
            data = np.moveaxis(np.array(list(mat['mocap'][0][0])), 1, 0)

            keypoints = list(mat['mocap'][0][0].dtype.fields.keys())
            indices_to_drop = [keypoints.index(k) for k in keypoints_to_drop]
            indices_to_keep = [i for i in np.arange(len(keypoints)) if i not in indices_to_drop]
            new_data = data[:, indices_to_keep, :]
            new_data = new_data.reshape(new_data.shape[0], -1)

            ## format into the csv for optipose
            keypoints_to_keep = np.array(keypoints)[indices_to_keep]
            columns = []
            for k in keypoints_to_keep:
                columns.extend([f'{k}_1', f'{k}_2', f'{k}_3'])

            count = 0
            # look for "holes" without any nans
            out = find_holes(mask=np.any(np.isnan(new_data), axis=1)[:, np.newaxis],
                             keypoints=['all'],
                             target_val=False,
                             min_size_hole=min_sample_length)
            for start, length, _ in out:
                df = pd.DataFrame(columns=columns, data=new_data[start:start+length])
                # df.loc[:, 'behaviour'] = np.nan
                csv_file = os.path.join(base_folder, f'{mf.split(".")[0]}_{count:04d}_dropkp.csv')
                df.to_csv(csv_file, index=False)
                csv_files.append(csv_file)
                count += 1

        print(len(csv_files))
        print(keypoints_to_keep, config.body_parts)
        data_stores = [FlattenedDataStore(config.body_parts, path) for path in csv_files]
        generate_dataset(output_folder, name, is_val, data_stores, n_samples, min_seq_length=min_sample_length,
                     max_seq_length=min_sample_length, min_x=-700,
                     max_x=700, min_y=-700, max_y=700, prefix="", suffix=suffix, random_noise=50)