import keypoint_moseq as kpms
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax_moseq.utils import unbatch
from jax_moseq.models.keypoint_slds import estimate_coordinates

import pandas as pd


if __name__ == '__main__':
    model_name = '2024_09_26-14_57_06'
    project_dir = 'demo_project'
    input_dir = '/projects/ag-bozek/france/results_behavior/datasets/INH_FL2_keypoints_1_60_wresiduals_w1nan_stride0.5_new'
    latent_dim = 11
    config = lambda: kpms.load_config(project_dir)
    bodyparts = ['left_hip', 'right_hip', 'left_coord', 'right_coord',
                 'left_back', 'right_back', 'left_knee', 'right_knee']
    skeleton = [['left_hip', 'right_hip'],
                ['left_back', 'right_back'],
                ['left_hip', 'left_coord'],
                ['left_knee', 'left_hip'],
                ['left_coord', 'left_back'],
                ['right_hip', 'right_coord'],
                ['right_knee', 'right_hip'],
                ['right_coord', 'right_back'],
                ]

    kpms.setup_project(
        project_dir,
        video_dir=input_dir,
        bodyparts=bodyparts,
        skeleton=skeleton,
        overwrite=True)

    kpms.update_config(project_dir,
                       video_dir=input_dir,
                       anterior_bodyparts=['left_back'],
                       posterior_bodyparts=['left_knee'],
                       use_bodyparts=bodyparts)

    # load data (e.g. from DeepLabCut)
    keypoint_data_path = os.path.join(input_dir,
                                      'train_dataset_w-1-nans.npz')  # can be a file, a directory, or a list of files
    coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_data_path, 'disk')

    # format data for modeling
    data, metadata = kpms.format_data(coordinates, confidences, **config())

    kpms.update_config(project_dir, latent_dim=latent_dim)

    num_ar_iters = 50

    # load model checkpoint
    model, data, metadata, current_iter = kpms.load_checkpoint(
        project_dir, model_name, iteration=num_ar_iters)

    # modify kappa to maintain the desired syllable time-scale
    model = kpms.update_hypparams(model, kappa=1e4)

    keypoint_data_path = os.path.join(input_dir,
                                      'test_fulllength_dataset_w-all-nans.npz')  # can be a file, a directory, or a list of files
    coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_data_path, 'disk')
    print('***', np.any(np.isnan(coordinates[list(coordinates.keys())[0]])), flush=True)
    print(coordinates[list(coordinates.keys())[0]].shape)
    print(np.unique(confidences[list(coordinates.keys())[0]]))

    data, metadata = kpms.format_data(coordinates, confidences, keys=None, **config())
    # data is a dictionary with 3 keys: Y, conf, mask
    # data['Y'] is of shape (10, 3630, 8, 3)
    # data['conf'] is of shape (10, 3630, 8)
    # data['mask'] is of shape (10, 3630)

    # apply saved model to new data
    results, applied_model = kpms.apply_model(model, data, metadata, project_dir, model_name, **config(),
                                              parallel_message_passing=False, return_model=True)

    # compute the estimated coordinates
    Y_est = estimate_coordinates(
        jnp.array(applied_model['states']['x']),
        # The pose trajectory is stored in the model as a variable “x” that encodes a low-dimensional representation of the keypoints (similar to PCA).
        jnp.array(applied_model['states']['v']),
        jnp.array(applied_model['states']['h']),
        jnp.array(applied_model['params']['Cd'])
    )
    print('Estimate coordinates after applying', Y_est.shape)
    # generate a dictionary with reconstructed coordinates for each recording
    # project back to original space?
    coordinates_est = unbatch(Y_est, *metadata)

    test_dir = '/projects/ag-bozek/france/results_behavior/outputs/25-09-24_FL2_new_for_comparison/DISK_test/test_for_optipose_repeat_0/'
    name = 'test'
    f = glob(os.path.join(test_dir, f'test_repeat-0_sample0.csv'))[0]
    to_save_columns = pd.read_csv(f).columns
    for k in list(coordinates.keys()):
        to_save = pd.DataFrame(data=coordinates_est[k].reshape(-1, 24), columns=to_save_columns)
        to_save.to_csv(os.path.join(test_dir, 'kpmoseq', f'test_repeat-0_file{k.split("track")[1]}_kpmoseq.csv'),
                       index=False)

    test_dir = '/projects/ag-bozek/france/results_behavior/outputs/25-09-24_FL2_new_for_comparison/DISK_test/test_for_optipose_repeat_0/'
    name = 'test'
    f = glob(os.path.join(test_dir, f'test_repeat-0_sample0.csv'))[0]

    results, applied_model = kpms.apply_model(model, data, metadata, project_dir, model_name, **config(),
                                              parallel_message_passing=False, return_model=True)
    to_save_columns = pd.read_csv(f).columns


    gt_df = pd.read_csv(os.path.join(test_dir, 'test_repeat-0.csv'), sep='|')
    # 'input' is with holes -> filled with -4668
    # 'label' is without holes -> ground truth
    coordinates_gt = {}
    coordinates = {}
    confidences = {}
    for i in range(len(gt_df)):
        coordinates_gt.update({f"{name}_track{i}": np.array(eval(gt_df.loc[i, 'label']))})
        f = glob(os.path.join(test_dir, f'test_repeat-0_sample{i}.csv'))[0]
        coords = pd.read_csv(f).values.reshape(-1, 8, 3)
        confs = (~np.isnan(coords[..., 0])).astype('float')
        coordinates.update({f"{name}_track{i}": coords})
        confidences.update({f"{name}_track{i}": confs})

    data, metadata = kpms.format_data(coordinates, confidences, keys=None, **config())
    # data is a dictionary with 3 keys: Y, conf, mask
    # data['Y'] is of shape (10, 3630, 8, 3)
    # data['conf'] is of shape (10, 3630, 8)
    # data['mask'] is of shape (10, 3630)

    # apply saved model to new data
    results, applied_model = kpms.apply_model(model, data, metadata, project_dir, model_name, **config(),
                                              parallel_message_passing=False, return_model=True)

    # compute the estimated coordinates
    Y_est = estimate_coordinates(
        jnp.array(applied_model['states']['x']),
        # The pose trajectory is stored in the model as a variable “x” that encodes a low-dimensional representation of the keypoints (similar to PCA).
        jnp.array(applied_model['states']['v']),
        jnp.array(applied_model['states']['h']),
        jnp.array(applied_model['params']['Cd'])
    )
    print('Estimate coordinates after applying', Y_est.shape)
    # generate a dictionary with reconstructed coordinates for each recording
    # project back to original space?
    coordinates_est = unbatch(Y_est, *metadata)

    for k in list(coordinates.keys()):
        to_save = pd.DataFrame(data=coordinates_est[k].reshape(-1, 24), columns=to_save_columns)
        to_save.to_csv(os.path.join(test_dir, 'kpmoseq', f'test_repeat-0_sample{k.split("track")[1]}_kpmoseq.csv'),
                       index=False)