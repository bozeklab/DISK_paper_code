import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from glob import glob

import matplotlib
if os.uname().nodename == 'france-XPS':
    matplotlib.use('TkAgg')
    basedir = '/home/france/Mounted_dir'
else:
    matplotlib.use('Agg')
    basedir = '/projects/ag-bozek/france'


def find_holes(mask, keypoints, target_val=1, indep=True, min_size_hole=60):
    """
    Find holes defined as equal to target_val in a 2D or 3D numpy array or pytorch tensor.

    Parameters
    ----------
    mask : numpy array or pytorch tensor of shape (time, keypoints, 3) or (time, keypoints)
    keypoints : list of names of keypoints, should have the same shape as mask.shape[1]
    target_val : when equal to target_val consider a hole
    indep : bool, considering the keypoints independently or as combination

    Returns out: a list of tuples (length_nan, keypoint_name)
    -------

    """
    # holes are where mask == target_val
    if len(mask.shape) == 2:
        mask = mask.reshape((mask.shape[0], len(keypoints), -1))
    module_ = np

    out = []
    if indep:
        for i_kp in range(len(keypoints)):
            # safer to loop on the keypoints, and process the mask 1D
            # probably slower
            start = 0
            mask_kp = mask[:, i_kp]
            while start < mask_kp.shape[0]:
                if not module_.any(mask_kp[start:] == target_val):
                    break
                index_start_nan = module_.where(mask_kp[start:] == target_val)[0][0]
                if module_.any(~(mask_kp[start + index_start_nan:] == target_val)):
                    length_nan = module_.where(~(mask_kp[start + index_start_nan:] == target_val))[0][0]
                else:
                    # the nans go until the end of the vector
                    length_nan = mask_kp.shape[0] - start - index_start_nan
                if length_nan >= min_size_hole:
                    out.append((start + index_start_nan, length_nan, keypoints[i_kp]))
                start = start + index_start_nan + length_nan

    return out  # returns a list of tuples (length_nan, keypoint_name)


if __name__ == '__main__':

    dataset_name = 'Fish_v3_60stride120'
    min_length = 60
    n_keypoints = 6
    #
    # dataset_name = 'Mocap_keypoints_60_stride30_new'
    # min_length = 60
    # n_keypoints = 20
    #
    # dataset_name = 'DF3D_keypoints_60stride5_new'
    # min_length = 60
    # n_keypoints = 38
    #
    # dataset_name = 'INH_CLB_keypoints_1_60_stride0.5'
    # min_length = 60
    # n_keypoints = 8
    #
    # dataset_name = 'DANNCE_seq_keypoints_60_stride30_fill10_new'
    # min_length = 60
    # n_keypoints = 20
    #
    # dataset_name = 'INH_FL2_keypoints_1_60_wresiduals_w1nan_stride0.5_new'
    # min_length = 60
    # n_keypoints = 8

    np_dataset_files = [('train', os.path.join(basedir, f'results_behavior/datasets/{dataset_name}/train_fulllength_dataset_w-all-nans.npz')),
                        ('val', os.path.join(basedir, f'results_behavior/datasets/{dataset_name}/val_fulllength_dataset_w-all-nans.npz'))]
    output_dir = os.path.join(basedir, f'results_behavior/datasets/{dataset_name}/for_optipose/')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)


    for part, f in np_dataset_files:
        count = 0
        if not os.path.exists(os.path.join(output_dir, part)):
            os.mkdir(os.path.join(output_dir, part))
        dataset = np.load(f)
        X = dataset['X']
        columns = []
        for k in range(n_keypoints):
            columns.extend([f'{k}_1', f'{k}_2', f'{k}_3'])

        print(X.shape)
        for x in tqdm(X):
            # look for "holes" without any nans
            out = find_holes(mask=np.any(np.isnan(x), axis=1)[:, np.newaxis],
                             keypoints=['all'],
                             target_val=False,
                             min_size_hole=min_length)
            for start, length, _ in out:
                df = pd.DataFrame(columns=columns, data=x[start:start+length])
                df.loc[:, 'behaviour'] = np.nan
                df.to_csv(os.path.join(output_dir, part, f'{dataset_name}_{count:03d}.csv'), index=False)
                count += 1


    # test file
    test_file = os.path.join(basedir, f'results_behavior/datasets/{dataset_name}/test_fulllength_dataset_w-all-nans.npz')
    dataset = np.load(test_file)
    X = dataset['X']
    columns = []
    for k in range(n_keypoints):
        columns.extend([f'{k}_1', f'{k}_2', f'{k}_3'])

    for i_x, x in tqdm(enumerate(X)):
        df = pd.DataFrame(columns=columns, data=x)
        df.to_csv(os.path.join(output_dir, f'{os.path.basename(test_file).split(".")[0]}_file-{i_x:03d}.csv'), index=False)
