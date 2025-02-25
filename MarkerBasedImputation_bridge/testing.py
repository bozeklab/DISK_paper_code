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

from build_ensemble import EnsembleModel
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


def open_data_csv(filepath, dataset_path, stride=20,
                  front_point=('left_coord', 'right_coord'), middle_point=('left_hip', 'right_hip'),
                  input_length=9, output_length=1):
    """Load data from csv file used for comparison."""

    dataset_constant_file = glob(os.path.join(dataset_path, 'constants.py'))[0]
    dataset_constants = read_constant_file(dataset_constant_file)
    n_keypoints = len(dataset_constants.KEYPOINTS)

    exclude_value = np.nan
    data = np.load(filepath)
    coords = data['X']

    transformed_coords, rot_angle, mean_position = preprocess_data(coords, dataset_constants.KEYPOINTS,
                                                                   middle_point=middle_point,
                                                                   front_point=front_point,
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
                    :dataset_constants.DIVIDER].reshape((ground_truth.shape[0], input_length + output_length, -1))

    transforms_dict = {'rot_angle': rot_angle,
                       'mean_position': mean_position,
                       'marker_means': marker_means,
                       'marker_stds': marker_stds,
                       'exclude_value': exclude_value}

    return input, ground_truth, dataset_constants, exclude_value, transforms_dict



def predict_markers(model, dict_model, X, bad_frames, keypoints, divider, ground_truth=None, markers_to_fix=None,
                    error_diff_thresh=.25, outlier_thresh=3, device=torch.device('cpu'), save_path='',
                    save_prefix='',
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

    # Preallocate
    processed_ground_truth = np.copy(ground_truth)
    preds = np.copy(X)
    pred = np.zeros((X.shape[0], output_length, X.shape[2]))

    # At each step, generate a prediction, replace the predictions of
    # markers you do not want to predict with the ground truth, and append
    # the resulting vector to the end of the next input chunk.
    while np.sum(mask) > 0 and np.max(startpoint[mask]) > - input_length:
        # print(set(zip(startpoint, next_frame_id)))

        # if first missing value before input_length, then pad before with first value
        processed_X_start = np.vstack([x[np.array([max(0, t) for t in range(s, s + input_length)])][np.newaxis] for (x, s) in zip(preds[mask], startpoint[mask])])

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
        if type(pred) == tuple: # this is the case if ensemble model
            pred = pred[0]
        pred = pred.detach().cpu().numpy()

        if ground_truth is not None:
            rmse = np.sqrt((pred[:, 0] - processed_ground_truth[mask, next_frame_id[mask]])**2)
            rmse[~bad_frames[mask, next_frame_id[mask]]] = np.nan
            rmse = np.nanmean(rmse, axis=1)

        if len(np.where(startpoint[mask] > 0)[0]) > 0:
            for item in np.random.choice(np.where(startpoint[mask] > 0)[0], 1):
                fig, axes = plt.subplots(pred.shape[-1] // divider, divider, figsize=(10, 10), sharey='col')
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
                    if i%divider == 0:
                        axes[i].set_ylabel(keypoints[i//divider])
                    if i==2:
                        axes[i].legend()
                if ground_truth is not None:
                    plt.suptitle(f'RMSE {rmse[item]:.3f}')
                plt.savefig(os.path.join(save_path, f'{save_prefix}_single_pred_single_step_item-{item}.png'))
                plt.close()

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
        plt.savefig(os.path.join(save_path, f'{save_prefix}_hist_rmse_values.png'))


    for item in np.random.choice(preds.shape[0], 10):

        fig, axes = plt.subplots(pred.shape[-1]//divider, divider, figsize=(10, 10), sharey='col', sharex='all')
        axes = axes.flatten()
        for i in range(pred.shape[-1]):
            t = np.arange(X.shape[1])
            if ground_truth is not None:
                axes[i].plot(t, processed_ground_truth[item, :, i], 'o-')
                plt.suptitle(f'RMSE = {rmse[item]:.3f}')
            else:
                x = X[item, :, i]
                x[get_mask(x, exclude_value)] = np.nan
                axes[i].plot(x, 'o-')
            axes[i].plot(t[bad_frames_orig[item, :, i]], preds[item, bad_frames_orig[item, :, i], i], 'x')
        plt.savefig(os.path.join(save_path, f'{save_prefix}_single_pred_item-{item}.png'))
        plt.close()

    transforms_dict = {} # no transforms, done in the loading phase

    return preds, bad_frames_orig, transforms_dict



def testing_single_model_like_predict(model_path, data_file, dataset_path, *,
                        save_path=None, front_point='', middle_point='',
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

    # Load model
    if model is None:
        logging.info(f'Loading ensemble model from {os.path.join(os.path.dirname(model_path), "training_info.json")}')
        with open(os.path.join(os.path.dirname(model_path), "training_info.json"), 'r') as fp:
            dict_training = json.load(fp)
        model = Wave_net(device=device, **dict_training)
        checkpoint = torch.load(os.path.join(basedir, model_path))
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
    else:
        raise ValueError(f'[testing_single_model_like_predict] model is None')

    markers, ground_truth, dataset_constants, exclude_value, transform_dict = open_data_csv(filepath=data_file,
                                                                                            dataset_path=dataset_path,
                                                                                            middle_point=middle_point,
                                                                                            front_point=front_point,
                                                                                            input_length=model.input_length,
                                                                                            output_length=model.output_length)
    bad_frames = get_mask(markers, exclude_value) # value used by optipose to mark dropped frames

    save_path_direction = None
    if save_path is not None:
        save_path_direction = os.path.join(save_path, f'{os.path.basename(data_file).split(".npz")[0]}_testing')
        if not os.path.exists(save_path_direction):
            os.mkdir(save_path_direction)

    preds, bad_frames, transform_dict = predict_markers(model, dict_training, markers, bad_frames,
                                                        dataset_constants.KEYPOINTS,
                                                        dataset_constants.DIVIDER, ground_truth,
                                                        save_path=save_path_direction,
                                                        save_prefix='testing_single_model_like_predict',
                                                        exclude_value=exclude_value)

    return preds


def testing_ensemble_model_like_predict(model_path, data_file, dataset_path, *,
                        save_path=None, middle_point='', front_point='',
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

    # Load model
    if model is None:
        logging.info(f'Loading ensemble model from {os.path.join(os.path.dirname(model_path), "training_info.json")}')
        with open(os.path.join(os.path.dirname(model_path), "training_info.json"), 'r') as fp:
            dict_training = json.load(fp)
        model = EnsembleModel(device=device, **dict_training)
        model.load_state_dict(torch.load(os.path.join(basedir, model_path)))
        model.eval()
    else:
        raise ValueError(f'[testing_ensemble_model_like_predict] model is None')

    markers, ground_truth, dataset_constants, exclude_value, transform_dict = open_data_csv(filepath=data_file,
                                                                                            dataset_path=dataset_path,
                                                                                            middle_point=middle_point,
                                                                                            front_point=front_point,
                                                                                            input_length=model.input_length,
                                                                                            output_length=model.output_length
                                                                                            )
    bad_frames = get_mask(markers, exclude_value) # value used by optipose to mark dropped frames

    save_path_direction = None
    if save_path is not None:
        save_path_direction = os.path.join(save_path, f'{os.path.basename(data_file).split(".npz")[0]}_testing')
        if not os.path.exists(save_path_direction):
            os.mkdir(save_path_direction)

    preds, bad_frames, transform_dict = predict_markers(model, dict_training, markers, bad_frames,
                                                        dataset_constants.KEYPOINTS,
                                                        dataset_constants.DIVIDER,
                                                        ground_truth,
                                                        save_path=save_path_direction,
                                                        save_prefix='testing_ensemble_model_like_predict',
                                                        exclude_value=exclude_value)

    return preds
