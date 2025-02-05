"""Imputes markers with mbi models."""
import numpy as np
import os
from scipy.io import savemat
import logging
from glob import glob
import torch
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import json

from models import Wave_net
from preprocess_data import preprocess_data, unprocess_data, apply_transform, z_score_data, apply_z_score
from utils import get_mask
from DISK.utils.utils import read_constant_file

if os.uname().nodename == 'france-XPS':
    matplotlib.use('TkAgg')
    basedir = '/home/france/Mounted_dir'
else:
    matplotlib.use('Agg')
    basedir = '/projects/ag-bozek/france'


def open_data_csv(filepath, dataset_path, stride=20):
    """Load data from csv file used for comparison."""

    dataset_constant_file = glob(os.path.join(dataset_path, 'constants.py'))[0]
    dataset_constants = read_constant_file(dataset_constant_file)
    n_keypoints = len(dataset_constants.KEYPOINTS)

    exclude_value = np.nan
    data = np.load(filepath)

    coords = data['X']
    input_length = 9
    output_length = 1

    transformed_coords, rot_angle, mean_position = preprocess_data(coords, dataset_constants.KEYPOINTS,
                                                                   middle_point=['left_hip', 'right_hip'],
                                                                   front_point=['left_coord', 'right_coord'],
                                                                   exclude_value=exclude_value)

    _, marker_means, marker_stds = z_score_data(transformed_coords.reshape(1, -1, transformed_coords.shape[2]),
                                                exclude_value=exclude_value)

    z_score_coords = apply_z_score(transformed_coords, marker_means, marker_stds, exclude_value)

    idx = np.arange(0, z_score_coords.shape[1] - (input_length + output_length), stride)
    input = np.vstack([[v[i: i + input_length + 1] for i in idx] for v in z_score_coords])
    input = input.reshape((input.shape[0], input_length + output_length, n_keypoints, -1))[..., :3].reshape(
        (input.shape[0], input_length + output_length, -1))
    input[:, -1] = exclude_value

    ground_truth = np.vstack(
        [[v[i: i + input_length + output_length][np.newaxis] for i in idx] for v in z_score_coords])
    ground_truth = ground_truth.reshape((ground_truth.shape[0], input_length + output_length, n_keypoints, -1))[...,
                    :3].reshape((ground_truth.shape[0], input_length + output_length, -1))


    transforms_dict = {'rot_angle': rot_angle,
                       'mean_position': mean_position,
                       'marker_means': marker_means,
                       'marker_stds': marker_stds,
                       'exclude_value': exclude_value}

    return input, ground_truth, dataset_constants, exclude_value, transforms_dict



def predict_markers(model, dict_model, X, bad_frames, keypoints, ground_truth=None, markers_to_fix=None,
                    error_diff_thresh=.25, outlier_thresh=3, device=torch.device('cpu'), save_path='',
                    exclude_value=-4668):
    """Imputes the position of missing markers.

    :param model: Ensemble model to use for prediction
    :param X: marker data (n_frames x n_markers)
    :param bad_frames: local matrix of shape == X.shape where 0 denotes a
                       tracked frame and 1 denotes a dropped frame
    :param fix_errors: boolean vector of length n_markers. True if you wish to
                       override marker on frames further than error_diff_thresh
                       from the previous prediction.
    :param error_diff_thresh: z-scored distance at which predictions override
                              marker measurements.
    :param outlier_thresh: Threshold at which to ignore model predictions.
    :param return_member_data: If true, also return the predictions of each
                               ensemble member in a matrix of size
                               n_members x n_frames x n_markers. The
    :return: preds, bad_frames
    """
    # Get the input lengths and find the first instance of all good frames.
    input_length = dict_model["input_length"]
    output_length = dict_model["output_length"]

    # See whether you should fix errors
    fix_errors = np.any(markers_to_fix)

    # Reshape and get the starting seed.
    # X has shape (samples, time, keypoints * 3D)
    bad_frames_any = np.any(bad_frames, axis=2) # axis=2 is the keypoint axis
    startpoint = np.argmax(bad_frames_any, axis=1) # returns the first point of missing = next_frame_id
    next_frame_id = startpoint
    startpoint = np.clip(startpoint - input_length, a_min=-input_length - 1, a_max=X.shape[1] - 1)
    mask = next_frame_id > 0 # only consider the samples needing imputation

    # if ground_truth is not None:
    #     processed_ground_truth, rot_angle_GT, mean_position_GT = preprocess_data(ground_truth,
    #                                                                              keypoints,
    #                                                                              middle_point=['right_hip', 'left_hip'],
    #                                                                              front_point=['right_coord', 'left_coord'],
    #                                                                              exclude_value=exclude_value)
    #
    # processed_X, rot_angle, mean_position = preprocess_data(X,
    #                                                               keypoints,
    #                                                               middle_point=['right_hip', 'left_hip'],
    #                                                               front_point=['right_coord', 'left_coord'],
    #                                                               exclude_value=exclude_value)
    #
    # _, marker_means, marker_stds = z_score_data(processed_X.reshape(1, -1, processed_X.shape[2]), exclude_value=exclude_value)
    #
    # if ground_truth is not None:
    #     processed_ground_truth = apply_z_score(processed_ground_truth, marker_means, marker_stds, exclude_value=exclude_value)

    processed_X = np.copy(X)
    processed_ground_truth = np.copy(ground_truth)

    # Preallocate
    preds = np.copy(processed_X)
    pred = np.zeros((X.shape[0], output_length, X.shape[2]))

    member_stds = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    # member_pred = np.zeros((X.shape[0], model.n_members, X.shape[2]))

    # At each step, generate a prediction, replace the predictions of
    # markers you do not want to predict with the ground truth, and append
    # the resulting vector to the end of the next input chunk.
    while np.sum(mask) > 0 and np.max(startpoint[mask]) > - input_length:
        # print(set(zip(startpoint, next_frame_id)))

        # if first missing value before input_length, then pad before with first value
        X_start = np.vstack([x[np.array([max(0, t) for t in range(s, s + input_length)])][np.newaxis] for (x, s) in zip(preds[mask], startpoint[mask])])

        processed_X_start = np.copy(X_start)
        # processed_X_start, rot_angle, mean_position = preprocess_data(X_start,
        #                                                         keypoints,
        #                                                         middle_point=['right_hip', 'left_hip'],
        #                                                         front_point=['right_coord', 'left_coord'],
        #                                                         exclude_value=exclude_value)
        # processed_X_start = apply_z_score(processed_X_start, marker_means, marker_stds, exclude_value=-4668)

        # If there is a marker prediction that is greater than the
        # difference threshold above, mark it as a bad frame.
        # These are likely just jumps or identity swaps from MoCap that
        # were not picked up by preprocessing.
        if fix_errors:
            diff = pred[:, 0, :] - X[:, next_frame_id, :]
            errors = np.squeeze(np.abs(diff) > error_diff_thresh)
            errors[~markers_to_fix] = False
            bad_frames[next_frame_id, errors] = True

        pred = model(torch.Tensor(processed_X_start).to(device))
        pred = pred.detach().cpu().numpy()

        if ground_truth is not None:
            rmse = np.sqrt((pred[:, 0] - processed_ground_truth[mask, next_frame_id[mask]])**2)
            rmse[~bad_frames[mask, next_frame_id[mask]]] = np.nan
            rmse = np.nanmean(rmse, axis=1)


        if len(np.where(startpoint[mask] > 0)[0]) > 0:
            for item in np.random.choice(np.where(startpoint[mask] > 0)[0], 1):
                fig, axes = plt.subplots(pred.shape[-1] // 3, 3, figsize=(10, 10), sharey='col')
                axes = axes.flatten()
                for i in range(pred.shape[-1]):
                    t = np.arange(9)
                    x = processed_X_start[item, :, i]
                    x[get_mask(x, exclude_value)] = np.nan
                    axes[i].plot(list(t) + [9], processed_ground_truth[mask][item][next_frame_id[mask][item]-9:next_frame_id[mask][item] + 1][:, i], 'o-', label='GT')
                    axes[i].plot(x, 'o-', label='input')
                    if bad_frames[mask, next_frame_id[mask]][item, i]:
                        axes[i].plot(9, pred[item, 0, i], 'v', c= 'red', label='pred wo missing data')
                    else:
                        axes[i].plot(9, pred[item, 0, i], 'v', c='cyan', label='pred w missing data')
                    if i%3 == 0:
                        axes[i].set_ylabel(keypoints[i//3])
                    if i==2:
                        axes[i].legend()
                if ground_truth is not None:
                    plt.suptitle(f'RMSE {rmse[item]:.3f}')
                plt.savefig(os.path.join(save_path, f'single_pred_single_step_item-{item}.png'))
                plt.close()
            print('stop')


        # Only use the predictions for the bad markers. Take the
        # predictions and append to the end of X_start for future
        # prediction.
        pred[:, 0][~bad_frames[mask, next_frame_id[mask]]] = preds[mask, next_frame_id[mask]][~bad_frames[mask, next_frame_id[mask]]]
        preds[mask, next_frame_id[mask]] = np.squeeze(pred)

        bad_frames = get_mask(preds, exclude_value)
        bad_frames_any = np.any(bad_frames, axis=2)  # axis=2 is the keypoint axis
        logging.info(f'Progress: remaining missing values = {np.sum(bad_frames_any)}')
        startpoint = np.argmax(bad_frames_any, axis=1)  # returns the first point of missing = next_frame_id
        next_frame_id = startpoint
        startpoint = np.clip(startpoint - input_length, a_min=-input_length - 1, a_max=X.shape[1] - 1)
        mask = next_frame_id > 0

    bad_frames_orig = get_mask(X, exclude_value)

    if ground_truth is not None:
        rmse = np.sqrt((pred - processed_ground_truth) ** 2)
        rmse[~bad_frames_orig] = np.nan
        rmse = np.nanmean(np.squeeze(rmse), axis=(1, 2))
        print(rmse.shape)

        plt.figure()
        plt.hist(rmse, bins=50)
        plt.yscale('log')
        plt.suptitle(f'mean RMSE: {np.mean(rmse):.3f} +/- {np.std(rmse):.3f}')
        plt.savefig(os.path.join(save_path, 'hist_rmse_values.png'))


    for item in np.random.choice(preds.shape[0], 10):

        fig, axes = plt.subplots(pred.shape[-1]//3, 3, figsize=(10, 10), sharey='col', sharex='all')
        axes = axes.flatten()
        for i in range(pred.shape[-1]):
            t = np.arange(X.shape[1])
            if ground_truth is not None:
                axes[i].plot(t, processed_ground_truth[item, :, i], 'o-')
                plt.suptitle(f'RMSE = {rmse[item]:.3f}')
            else:
                x = processed_X[item, :, i]
                x[get_mask(x, exclude_value)] = np.nan
                axes[i].plot(x, 'o-')
            axes[i].plot(t[bad_frames_orig[item, :, i]], preds[item, bad_frames_orig[item, :, i], i], 'x')
        plt.savefig(os.path.join(save_path, f'single_pred_item-{item}.png'))
        plt.close()

    transforms_dict = {}
    # {'rot_angle': rot_angle,
    #                    'mean_position': mean_position,
    #                    'marker_means': marker_means,
    #                    'marker_stds': marker_stds}

    return preds, bad_frames_orig, member_stds, transforms_dict



def testing(model_path, data_file, dataset_path, *,
                        save_path=None, stride=1, n_folds=10, fold_id=0,
                        markers_to_fix=None, error_diff_thresh=.25,
                        model=None, device=torch.device('cpu')):
    """Imputes the position of missing markers.

    :param model_path: Path to model to use for prediction.
    :param data_path: Path to marker and bad_frames data. csv.
    :param pass_direction: Direction in which to impute markers.
                           Can be 'forward' or 'reverse'
    :param save_path: Path to a folder in which to store the prediction chunks.
    :param stride: stride length between frames for faster imputation.
    :param n_folds: Number of folds across which to divide data for faster
                    imputation.
    :param fold_id: Fold identity for this specific session.
    :param markers_to_fix: Markers for which to override suspicious MoCap
                           measurements
    :param error_diff_thresh: Z-scored difference threshold marking suspicious
                              frames
    :param model: Model to be used in prediction. Overrides model_path.
    :return: preds
    """

    markers, ground_truth, dataset_constants, exclude_value, transform_dict = open_data_csv(filepath=data_file, dataset_path=dataset_path)
    bad_frames = get_mask(markers, exclude_value) # value used by optipose to mark dropped frames

    # Load model
    if model is None:
        logging.info(f'Loading ensemble model from {os.path.join(os.path.dirname(model_path), "training_info.json")}')
        with open(os.path.join(os.path.dirname(model_path), "training_info.json"), 'r') as fp:
            dict_training = json.load(fp)
        model = Wave_net(device=device, **dict_training)
        model.load_state_dict(torch.load(os.path.join(basedir, model_path)))
        model.eval()

    # print('Predicting %d frames starting at frame %d.'
    #       % (markers.shape[1], start_frame))

    # If the model can return the member predictions, do so.
    save_path_direction = None
    if save_path is not None:
        save_path_direction = os.path.join(save_path, f'{os.path.basename(data_file).split(".npz")[0]}_testing')
        if not os.path.exists(save_path_direction):
            os.mkdir(save_path_direction)

    preds, bad_frames, member_stds, transform_dict = predict_markers(model, dict_training, markers, bad_frames,
                                                            dataset_constants.KEYPOINTS, ground_truth,
                                                            markers_to_fix=markers_to_fix,
                                                            error_diff_thresh=error_diff_thresh,
                                                            save_path=save_path_direction,
                                                            exclude_value=exclude_value)


    # Save predictions to a matlab file.
    if save_path is not None:
        # file_name = '%s_fold_id_%d.mat' % (pass_direction, fold_id)
        file_name = f'{os.path.basename(data_file).split(".csv")[0]}_testing.mat'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, file_name)
        logging.info(f'Saving to {save_path}')
        output_dict = {'preds': preds,
                        'markers': markers, # inputs to the model, z-scored
                       'marker_names': dataset_constants.KEYPOINTS,
                        'bad_frames': bad_frames, # original to keep track of where were the holes
                        # 'n_folds': n_folds,
                        # 'fold_id': fold_id,
                       'data_file': data_file,
                        'marker_means': transform_dict["marker_means"],
                        'marker_stds': transform_dict["marker_stds"],
                        'rot_angle': transform_dict["rot_angle"],
                        'mean_position': transform_dict["mean_position"],
                        'exclude_value': exclude_value,
                        'ground_truth': ground_truth,
                        'member_stds': member_stds}
        savemat(save_path, output_dict)

    return preds
