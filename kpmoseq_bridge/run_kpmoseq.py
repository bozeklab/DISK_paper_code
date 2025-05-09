import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import os
import numpy as np
import argparse
import pandas as pd
import logging
import importlib

import keypoint_moseq as kpms
import jax.numpy as jnp
from jax_moseq.utils import unbatch
from jax_moseq.models.keypoint_slds import estimate_coordinates
from jax_moseq.utils import set_mixed_map_iters
set_mixed_map_iters(4) ## batch size
import jax
jax.config.read('jax_enable_x64') # double precision

import matplotlib
if os.uname().nodename == 'france-XPS':
    matplotlib.use('TkAgg')
    basedir = '/home/france/Mounted_dir'
else:
    matplotlib.use('Agg')
    basedir = '/projects/ag-bozek/france'


def read_constant_file(constant_file):
    """import constant file as a python file from its path"""
    spec = importlib.util.spec_from_file_location("module.name", constant_file)
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)

    try:
        constants.NUM_FEATURES, constants.DIVIDER, constants.KEYPOINTS, constants.SEQ_LENGTH
    except NameError:
        print('constant file should have following keys: NUM_FEATURES, DIVIDER, KEYPOINTS, SEQ_LENGTH')
    constants.N_KEYPOINTS = len(constants.KEYPOINTS)

    return constants

def read_skeleton_file(skeleton_file, keypoints):
    """read skeleton file and returns skeleton well formatted"""
    spec = importlib.util.spec_from_file_location("module.name", skeleton_file)
    skeleton_inputs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(skeleton_inputs)

    neighbor_link = []
    for i in range(len(skeleton_inputs.neighbor_links)):
        if type(skeleton_inputs.neighbor_links[i][0]) == tuple:
            for nn in skeleton_inputs.neighbor_links[i]:
                neighbor_link.append([keypoints[nn[0]], keypoints[nn[1]]])
        else:
            neighbor_link.append([keypoints[skeleton_inputs.neighbor_links[i][0]],
                                  keypoints[skeleton_inputs.neighbor_links[i][1]]])
    return neighbor_link


if __name__ == '__main__':

    ##########################################################################################################
    ### CHOOSE DATASET BY SUPPLYING THE COMMANDLINE ARGUMENT
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset', type=str,
                        help='dataset name', choices=['FL2', 'CLB', 'DANNCE', 'Mocap', 'DF3D', 'Fish', 'MABe'])

    parser.add_argument('--train', '-t', action='store_true', default=False,
                        help='retrain')

    args = parser.parse_args()

    ##########################################################################################################
    ## ARGUMENTS FOR EACH DATASET

    ## FL2
    if args.dataset == 'FL2':
        project_dir = 'kpmoseq_FL2'
        input_dir = os.path.join(basedir, 'results_behavior/datasets/INH_FL2_keypoints_1_60_wresiduals_w1nan_stride0.5_new')
        anterior_bodyparts = ['left_back']
        posterior_bodyparts = ['left_knee']
        test_dir = os.path.join(basedir, 'results_behavior/outputs/25-09-24_FL2_new_for_comparison/DISK_test/test_for_optipose_repeat_0/')
        _2D = False

    ## DANNCE
    elif args.dataset == 'DANNCE':
        project_dir = 'kpmoseq_DANNCE'
        input_dir = os.path.join(basedir, 'results_behavior/datasets/DANNCE_seq_keypoints_60_stride30_fill10_new')
        anterior_bodyparts = ['SpineF']
        posterior_bodyparts = ['SpineM']
        test_dir = os.path.join(basedir, 'results_behavior/outputs/13-02-25_DANNCE_for_comparison/DISK_test/test_for_optipose_repeat_0/')
        _2D = False

    ## CLB
    elif args.dataset == 'CLB':
        project_dir = 'kpmoseq_CLB'
        input_dir = os.path.join(basedir, 'results_behavior/datasets/INH_CLB_keypoints_1_60_stride0.5')
        anterior_bodyparts = ['left_back']
        posterior_bodyparts = ['left_knee']
        test_dir = os.path.join(basedir, 'results_behavior/outputs/13-02-25_CLB_for_comparison/DISK_test/test_for_optipose_repeat_0/')
        _2D = False

    ## Mocap
    elif args.dataset == 'Mocap':
        project_dir = 'kpmoseq_Mocap'
        input_dir = os.path.join(basedir, 'results_behavior/datasets/Mocap_keypoints_60_stride30_new')
        anterior_bodyparts = ['arm1_1']
        posterior_bodyparts = ['arm1_0']
        test_dir = os.path.join(basedir, 'results_behavior/outputs/2025-02-24_Mocap_for_comparison/DISK_test/test_for_optipose_repeat_0/')
        _2D = False

    elif args.dataset == 'Fish':
        project_dir = 'kpmoseq_Fish'
        input_dir = os.path.join(basedir, 'results_behavior/datasets/Fish_v3_60stride120')
        anterior_bodyparts = ['fish1_head']
        posterior_bodyparts = ['fish2_tail']
        test_dir = os.path.join(basedir, 'results_behavior/outputs/2023-09-27_Fishv3_newnewmissing/DISK_test_for_comparison/test_for_optipose_repeat_0/')
        _2D = False

    elif args.dataset == 'MABe':
        project_dir = 'kpmoseq_MABe'
        input_dir = os.path.join(basedir, 'results_behavior/datasets/MABE_task1_60stride60')
        anterior_bodyparts = ['kp3_animal0']
        posterior_bodyparts = ['kp6_animal1']
        test_dir = os.path.join(basedir,
                                'results_behavior/outputs/2024-02-19_MABe_task1_newnewmissing/DISK_test/test_for_optipose_repeat_0/')
        _2D = True

    elif args.dataset == 'DF3D':
        project_dir = 'kpmoseq_DF3D'
        input_dir = os.path.join(basedir, 'results_behavior/datasets/DF3D_keypoints_60stride5_new')
        anterior_bodyparts = ['15']
        posterior_bodyparts = ['35']
        test_dir = os.path.join(basedir,
                                'results_behavior/outputs/2025-02-13_DF3D_for_comparison/DISK_test/test_for_optipose_repeat_0/')
        _2D = False


    ##########################################################################################################
    dataset_constant_file = glob(os.path.join(input_dir, 'constants.py'))[0]
    dataset_constants = read_constant_file(dataset_constant_file)
    bodyparts = dataset_constants.KEYPOINTS
    skeleton_file = glob(os.path.join(input_dir, 'skeleton.py'))[0]
    skeleton = read_skeleton_file(skeleton_file, bodyparts)
    print('bodypart:', bodyparts)
    print('skeleton:', skeleton)

    config = lambda: kpms.load_config(project_dir)

    kpms.setup_project(
        project_dir,
        video_dir=input_dir,
        bodyparts=bodyparts,
        skeleton=skeleton,
        overwrite=True)

    kpms.update_config(project_dir,
                       video_dir=input_dir,
                       anterior_bodyparts=anterior_bodyparts,
                       posterior_bodyparts=posterior_bodyparts,
                       use_bodyparts=bodyparts)

    ################## TRAIN #################
    ## load data (e.g. from DeepLabCut)
    train_file = 'train_fulllength_dataset_w-all-nans.npz'
    keypoint_data_path = os.path.join(input_dir, train_file)  # can be a file, a directory, or a list of files
    coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_data_path, 'disk')
    # transforms = init_transforms(viewinvariant=True, normalizecube=True, divider=3, outputdir=project_dir, length_input_seq=60)
    # coordinates = {}
    # for k in raw_coordinates.keys():
    #     coordinates[k] = transform_x(raw_coordinates[k], transforms)[0]
    num_ar_iters = 50

    if args.train:
        ## format data for modeling
        data, metadata = kpms.format_data(coordinates, confidences, **config())
        print('-- Initial data', data[list(data.keys())[0]].shape)

        ## only for 2D
        if _2D:
            kpms.noise_calibration(project_dir, coordinates, confidences, **config())

        pca = kpms.fit_pca(**data, **config())
        kpms.save_pca(pca, project_dir)

        f_pca = 0.9
        kpms.print_dims_to_explain_variance(pca, f_pca)
        cs = np.cumsum(pca.explained_variance_ratio_)
        if cs[-1] < f_pca:
            latent_dim = len(cs)
        else:
            latent_dim = int((cs>f_pca).nonzero()[0].min()+1)
        print(type(latent_dim), latent_dim)

        # kpms.plot_scree(pca, project_dir=project_dir)
        # kpms.plot_pcs(pca, project_dir=project_dir, **config())

        kpms.update_config(project_dir, latent_dim=latent_dim)

        ## initialize the model
        model = kpms.init_model(data, pca=pca, **config())

        _, model_name = kpms.fit_model(
            model, data, metadata, project_dir,
            ar_only=True, num_iters=num_ar_iters, parallel_message_passing=False)

        ## modify kappa to maintain the desired syllable time-scale
        model = kpms.update_hypparams(model, kappa=1e4)

        model, data, metadata, current_iter = kpms.load_checkpoint(
            project_dir, model_name, iteration=num_ar_iters)

        ## run fitting for an additional 500 iters
        model = kpms.fit_model(
            model, data, metadata, project_dir, model_name, ar_only=False,
            start_iter=current_iter, num_iters=current_iter + 500,
            parallel_message_passing=False)[0]
    else:
        model_name_list = glob(os.path.join(project_dir, '2025_*'))
        print(project_dir, model_name_list)
        model_name = os.path.basename(max(model_name_list, key=os.path.getctime))
        ## load model checkpoint
        model, data, metadata, current_iter = kpms.load_checkpoint(
            project_dir, model_name, iteration=500 + num_ar_iters)

    Y_est = estimate_coordinates(
        jnp.array(model['states']['x']),
        # The pose trajectory is stored in the model as a variable “x” that encodes a low-dimensional representation of the keypoints (similar to PCA).
        jnp.array(model['states']['v']),
        jnp.array(model['states']['h']),
        jnp.array(model['params']['Cd'])
    )
    print('Estimate coordinates after fitting', Y_est.shape)

    ################## TEST FULL LENGTH #################
    keypoint_data_path = os.path.join(input_dir, 'test_fulllength_dataset_w-all-nans.npz')  # can be a file, a directory, or a list of files
    coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_data_path, 'disk')
    print('***', np.any(np.isnan(coordinates[list(coordinates.keys())[0]])), flush=True)
    print(coordinates[list(coordinates.keys())[0]].shape)
    print(np.unique(confidences[list(coordinates.keys())[0]]))

    data, metadata = kpms.format_data(coordinates, confidences, keys=None, **config())
    ## data is a dictionary with 3 keys: Y, conf, mask

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
    #
    name = 'test'
    output_dir = os.path.join(test_dir, 'kpmoseq')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    f = glob(os.path.join(test_dir, f'test_repeat-0_sample0.csv'))[0]
    to_save_columns = pd.read_csv(f).columns
    for k in list(coordinates.keys()):
        to_save = pd.DataFrame(data=coordinates_est[k].reshape(-1, len(dataset_constants.KEYPOINTS) * dataset_constants.DIVIDER), columns=to_save_columns)
        to_save.to_csv(os.path.join(output_dir, f'test_repeat-0_file{k.split("track")[1]}_kpmoseq.csv'),
                       index=False)


    ################## TEST SHORT SEQUENCES #################
    gt_df = pd.read_csv(os.path.join(test_dir, 'test_repeat-0.csv'), sep='|')
    # 'input' is with holes -> filled with -4668
    # 'label' is without holes -> ground truth
    coordinates_gt = {}
    coordinates = {}
    confidences = {}
    for i in range(len(gt_df)):
        coordinates_gt.update({f"{name}_track{i}": np.array(eval(gt_df.loc[i, 'label']))})
        f = glob(os.path.join(test_dir, f'test_repeat-0_sample{i}.csv'))[0]
        coords = pd.read_csv(f).values.reshape(-1, len(dataset_constants.KEYPOINTS), dataset_constants.DIVIDER)
        confs = (~np.isnan(coords[..., 0])).astype('float')
        coordinates.update({f"{name}_track{i}": coords})
        confidences.update({f"{name}_track{i}": confs})
    #
    # keypoint_data_path = ''
    # # keypoint_data_path = os.path.join(input_dir, 'test_fulllength_dataset_w-1-nans.npz')  # can be a file, a directory, or a list of files
    # # coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_data_path, 'disk')
    # # print('***', np.any(np.isnan(coordinates[list(coordinates.keys())[0]])), flush=True)
    # # print(coordinates[list(coordinates.keys())[0]].shape)
    # # print(np.unique(confidences[list(coordinates.keys())[0]]))
    # # coordinates = {}
    # # for k in raw_coordinates.keys():
    # #     coordinates[k] = transform_x(raw_coordinates[k], transforms)[0]
    #
    # data, metadata = kpms.format_data(coordinates, confidences, keys=None, **config())
    # # data is a dictionary with 3 keys: Y, conf, mask
    # # data['Y'] is of shape (10, 3630, 8, 3)
    # # data['conf'] is of shape (10, 3630, 8)
    # # data['mask'] is of shape (10, 3630)
    #
    # # apply saved model to new data
    data, metadata = kpms.format_data(coordinates, confidences, keys=None, **config())
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


    for k in list(coordinates_est.keys()):
        print(coordinates_est[k].reshape(-1, len(dataset_constants.KEYPOINTS) * dataset_constants.DIVIDER).shape)
        to_save = pd.DataFrame(data=coordinates_est[k].reshape(-1, len(dataset_constants.KEYPOINTS) * dataset_constants.DIVIDER), columns=to_save_columns)
        to_save.to_csv(os.path.join(output_dir, f'test_repeat-0_sample{k.split("track")[1]}_kpmoseq.csv'),
                       index=False)