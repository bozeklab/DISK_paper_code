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
from preprocess_data import preprocess_data, unprocess_data, apply_transform, z_score_data, apply_z_score
from utils import get_mask
from DISK.utils.utils import read_constant_file

if os.uname().nodename == 'france-XPS':
    matplotlib.use('TkAgg')
    basedir = '/home/france/Mounted_dir'
else:
    matplotlib.use('Agg')
    basedir = '/projects/ag-bozek/france'


def open_data_csv(filepath, dataset_path, stride=20, front_point='', middle_point=''):
    """Load data from csv file used for comparison."""

    dataset_constant_file = glob(os.path.join(dataset_path, 'constants.py'))[0]
    dataset_constants = read_constant_file(dataset_constant_file)
    divider = dataset_constants.DIVIDER
    n_keypoints = len(dataset_constants.KEYPOINTS)
    if 'npz' in filepath:
        exclude_value = np.nan
        data = np.load(filepath)

        coords = data['X']
        input_length = 9
        output_length = 1

        # transformed_coords, rot_angle, mean_position = preprocess_data(coords, dataset_constants.KEYPOINTS,
        #                                                                middle_point=middle_point,
        #                                                                front_point=front_point,
        #                                                                exclude_value=exclude_value)
        #
        # _, marker_means, marker_stds = z_score_data(transformed_coords.reshape(1, -1, transformed_coords.shape[2]),
        #                                             exclude_value=exclude_value)
        #
        # z_score_coords = apply_z_score(transformed_coords, marker_means, marker_stds, exclude_value)

        idx = np.arange(0, coords.shape[1] - (input_length + output_length), stride)
        input = np.vstack([[v[i: i + input_length + 1] for i in idx] for v in coords])
        input = input.reshape((input.shape[0], input_length + output_length, n_keypoints, -1))[..., :divider].reshape(
            (input.shape[0], input_length + output_length, -1))
        input[:, -1] = exclude_value

        ground_truth = np.vstack(
            [[v[i: i + input_length + output_length][np.newaxis] for i in idx] for v in coords])
        ground_truth = ground_truth.reshape((ground_truth.shape[0], input_length + output_length, n_keypoints, -1))[...,
                        :divider].reshape((ground_truth.shape[0], input_length + output_length, -1))

    else:
        if 'repeat' in os.path.basename(filepath):
            df = pd.read_csv(filepath, sep='|')
            logging.info(f'input df columns: {df.columns}')
            input = np.vstack([np.array(eval(v))[np.newaxis] for v in df['input']])
            input = input.reshape(input.shape[0], input.shape[1], -1)
            ground_truth = np.vstack([np.array(eval(v))[np.newaxis] for v in df['label']])
            ground_truth = ground_truth.reshape(input.shape[0], input.shape[1], -1)

            exclude_value = -4668
        else:
            logging.info(f'-- filepath {filepath}')
            df = pd.read_csv(filepath, sep=',')
            cols = df.columns

            input = df[cols].values[np.newaxis]
            ground_truth = None
            logging.info(f'input shape: {input.shape}')
            exclude_value = np.nan

    processed_ground_truth = None
    if ground_truth is not None:
        processed_ground_truth, rot_angle_GT, mean_position_GT = preprocess_data(ground_truth,
                                                                                 dataset_constants.KEYPOINTS,
                                                                                 dataset_constants.DIVIDER,
                                                                                 middle_point=middle_point,
                                                                                 front_point=front_point,
                                                                                 exclude_value=exclude_value)

    processed_X, rot_angle, mean_position = preprocess_data(input,
                                                                  dataset_constants.KEYPOINTS,
                                                            dataset_constants.DIVIDER,
                                                                  middle_point=middle_point,
                                                                  front_point=front_point,
                                                                  exclude_value=exclude_value)

    _, marker_means, marker_stds = z_score_data(processed_X.reshape(1, -1, processed_X.shape[2]), exclude_value=exclude_value)
    processed_X = apply_z_score(processed_X, marker_means, marker_stds, exclude_value=exclude_value)

    if ground_truth is not None:
        processed_ground_truth = apply_z_score(processed_ground_truth, marker_means, marker_stds, exclude_value=exclude_value)

    transforms_dict = {'rot_angle': rot_angle,
                       'mean_position': mean_position,
                       'marker_means': marker_means,
                       'marker_stds': marker_stds}

    data = {'processed_X': processed_X,
            'processed_ground_truth': processed_ground_truth,
            'X': input,
            'ground_truth': ground_truth}

    return data, dataset_constants, exclude_value, transforms_dict


def predict_markers(model, dict_model, X, bad_frames, keypoints, divider, ground_truth=None,
                    device=torch.device('cpu'), save_path='',
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

    # # See whether you should fix errors
    # fix_errors = np.any(markers_to_fix)

    # Reshape and get the starting seed.
    # X has shape (samples, time, keypoints * 3D)
    bad_frames_any = np.any(bad_frames, axis=2) # axis=2 is the keypoint axis
    startpoint = np.argmax(bad_frames_any, axis=1) # returns the first point of missing = next_frame_id
    next_frame_id = startpoint
    startpoint = np.clip(startpoint - input_length, a_min=-input_length - 1, a_max=X.shape[1] - 1)
    mask = next_frame_id > 0 # only consider the samples needing imputation

    bad_frames_orig = get_mask(X, exclude_value)

    # Preallocate
    preds = np.copy(X)
    pred = np.zeros((X.shape[0], output_length, X.shape[2]))
    member_stds = np.zeros((X.shape[0], X.shape[1], X.shape[2]))

    # At each step, generate a prediction, replace the predictions of
    # markers you do not want to predict with the ground truth, and append
    # the resulting vector to the end of the next input chunk.
    print(save_path)
    print(np.sum(mask), startpoint, bad_frames_any[:, 0])
    n_iteration = 0
    while np.sum(mask) > 0 and np.max(startpoint[mask]) > - input_length:
        # if first missing value before input_length, then pad before with first value
        X_start = np.vstack([x[np.array([max(0, t) for t in range(s, s + input_length)])][np.newaxis] for (x, s) in zip(preds[mask], startpoint[mask])])

        pred, member_pred = model(torch.Tensor(X_start).to(device))
        pred = pred.detach().cpu().numpy()
        member_pred = member_pred.detach().cpu().numpy()
        # Detect anomalous predictions.
        # outliers = np.squeeze(np.abs(pred) > outlier_thresh)
        # pred[:, 0][outliers] = preds[mask, next_frame_id[mask]][outliers]
        if ground_truth is not None:
            rmse = np.sqrt((pred[:, 0] - ground_truth[mask, next_frame_id[mask]])**2)
            rmse[~bad_frames[mask, next_frame_id[mask]]] = np.nan
            rmse = np.nanmean(rmse, axis=1)

            # rmse_per_member = np.sqrt((member_pred[:, :, 0] - ground_truth[mask, next_frame_id[mask]][:, np.newaxis])**2)
            # for i_member in range(model.n_members):
            #     rmse_per_member[:, i_member][~bad_frames[mask, next_frame_id[mask]]] = np.nan
            # rmse_per_member = np.nanmean(rmse_per_member, axis=-1)

        if n_iteration < 25 and len(np.where(startpoint[mask] > 0)[0]) > 0:
            np.random.seed(42)
            for item in np.random.choice(len(startpoint), 5):
                if startpoint[item] > 0:
                    index_inside_mask = np.cumsum(mask[:item])[-1] if item > 0 else 0
                    fig, axes = plt.subplots(pred.shape[-1] // divider, divider, figsize=(10, 10), sharey='col')
                    axes = axes.flatten()
                    for i in range(pred.shape[-1]):
                        t = np.arange(9)
                        x = X_start[index_inside_mask, :, i]
                        x[get_mask(x, exclude_value)] = np.nan
                        if ground_truth is not None:
                            axes[i].plot(list(t) + [9], ground_truth[item][next_frame_id[item]-9:next_frame_id[item] + 1][:, i], 'o-', label='GT')
                        orig_bad_frames_mask = bad_frames_orig[item, next_frame_id[item]-9:next_frame_id[item], i]
                        pl = axes[i].plot(t[~orig_bad_frames_mask], x[~orig_bad_frames_mask], 'o-', label='input')
                        axes[i].plot(t[orig_bad_frames_mask], x[orig_bad_frames_mask], 'x--', label='input', c=pl[0].get_color())
                        if bad_frames[mask, next_frame_id[mask]][index_inside_mask, i]:
                            axes[i].plot(9, pred[index_inside_mask, 0, i], 'v', c= 'red', label='pred wo missing data')
                        else:
                            axes[i].plot(9, pred[index_inside_mask, 0, i], 'v', c='cyan', label='pred w missing data')
                        if i%divider == 0:
                            axes[i].set_ylabel(keypoints[i//divider])
                        if i==2:
                            axes[i].legend()
                    if ground_truth is not None:
                        plt.suptitle(f'RMSE {rmse[index_inside_mask]:.3f}')
                    plt.savefig(os.path.join(save_path, f'single_pred_single_step_item-{item}_iteration-{n_iteration}.png'))
                    plt.close()

        # Only use the predictions for the bad markers. Take the
        # predictions and append to the end of X_start for future
        # prediction.
        for i_member in range(0, model.n_members):
            member_pred[:, i_member, 0][~bad_frames[mask, next_frame_id[mask]]] = float('nan')
        member_std = np.nanstd(member_pred, axis=1)

        if np.all(np.isnan(member_std)):
            logging.info(f'NANs: {bad_frames[mask, next_frame_id[mask]][::divider].shape}, {np.any(bad_frames[mask, next_frame_id[mask]][::divider], axis=-1)}')

        for item in np.where(mask)[0]:
            index_inside_mask = np.cumsum(mask[:item])[-1] if item > 0 else 0
            preds[item, next_frame_id[item], bad_frames[item, next_frame_id[item]]] = np.squeeze(pred[index_inside_mask])[bad_frames[item, next_frame_id[item]]]

        member_stds[mask, next_frame_id[mask], :] = np.squeeze(member_std)

        if n_iteration < 25 and ground_truth is not None:
            rmse = np.sqrt((np.squeeze(pred) - ground_truth[mask, next_frame_id[mask]]) ** 2)
            rmse[~bad_frames_orig[mask, next_frame_id[mask]]] = np.nan
            rmse = np.nanmean(rmse, axis=1)
            print(rmse.shape)

            plt.figure()
            plt.hist(rmse, bins=50)
            plt.yscale('log')
            plt.suptitle(f'mean RMSE: {np.mean(rmse):.3f} +/- {np.std(rmse):.3f}')
            plt.savefig(os.path.join(save_path, f'hist_rmse_values_iteration-{n_iteration}.png'))
            plt.close()

        bad_frames = get_mask(preds, exclude_value)
        bad_frames_any = np.any(bad_frames, axis=2)  # axis=2 is the keypoint axis
        logging.info(f'Progress {n_iteration}: remaining missing values = {np.sum(bad_frames_any)}')
        startpoint = np.argmax(bad_frames_any, axis=1)  # returns the first point of missing = next_frame_id
        next_frame_id = startpoint
        startpoint = np.clip(startpoint - input_length, a_min=-input_length - 1, a_max=X.shape[1] - 1)
        mask = next_frame_id > 0

        n_iteration += 1

    if ground_truth is not None:
        rmse = np.sqrt((preds - ground_truth) ** 2)
        rmse[~bad_frames_orig] = np.nan
        rmse = np.nanmean(rmse, axis=(1, 2))
        print(rmse.shape)

        plt.figure()
        plt.hist(rmse, bins=50)
        plt.yscale('log')
        plt.suptitle(f'mean RMSE: {np.mean(rmse):.3f} +/- {np.std(rmse):.3f}')
        plt.savefig(os.path.join(save_path, f'hist_rmse_values_global.png'))
        plt.close()

    for item in np.random.choice(preds.shape[0], 10):
        figsize0 = 10 if X.shape[1] < 100 else 30
        fig, axes = plt.subplots(pred.shape[-1]//divider, divider, figsize=(figsize0, 10), sharey='col', sharex='all')
        axes = axes.flatten()
        for i in range(pred.shape[-1]):
            t = np.arange(X.shape[1])
            if ground_truth is not None:
                axes[i].plot(t, ground_truth[item, :, i], 'o-')
                plt.suptitle(f'RMSE = {rmse[item]:.3f}')
            else:
                x = X[item, :, i]
                x[get_mask(x, exclude_value)] = np.nan
                axes[i].plot(x, 'o-')
            axes[i].plot(t[bad_frames_orig[item, :, i]], preds[item, bad_frames_orig[item, :, i], i], 'x')
        plt.savefig(os.path.join(save_path, f'single_pred_item-{item}.png'))
        plt.close()

    return preds, bad_frames_orig, member_stds


def predict_single_pass(model_path, data_file, dataset_path, pass_direction, *,
                        save_path=None, stride=1,
                        front_point='', middle_point='',
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
    if not (pass_direction == 'forward') | (pass_direction == 'reverse'):
        raise ValueError('pass_direction must be forward or reverse')

    data, dataset_constants, exclude_value, transform_dict = open_data_csv(filepath=data_file,
                                                                                            dataset_path=dataset_path,
                                                                                            stride=stride,
                                                                                            front_point=front_point,
                                                                                            middle_point=middle_point)
    bad_frames = get_mask(data['X'], exclude_value) # value used by optipose to mark dropped frames

    # Load model
    if model is None:
        logging.info(f'Loading ensemble model from {os.path.join(os.path.dirname(model_path), "training_info.json")}')
        with open(os.path.join(os.path.dirname(model_path), "training_info.json"), 'r') as fp:
            dict_training = json.load(fp)
        model = EnsembleModel(device=device, **dict_training)
        model.load_state_dict(torch.load(os.path.join(basedir, model_path)))
        model.eval()

    # # Set Markers to fix
    # if markers_to_fix is None:
    #     markers_to_fix = np.zeros((markers.shape[1])) > 1
    #     # TODO(Skeleton): Automate this by including the skeleton.
    #     # Fix all arms, elbows, shoulders, shins, hips and legs.
    #     markers_to_fix[30:] = True
    #     # markers_to_fix[30:36] = True
    #     # markers_to_fix[42:] = True

    if pass_direction == 'reverse':
        markers = data['processed_X'][:, ::-1, :]
        ground_truth = data['processed_ground_truth'][:, ::-1, :] if data['processed_ground_truth'] is not None else None
        print(bad_frames[:, -1, 0])
        bad_frames = bad_frames[:, ::-1, :]
        print(bad_frames[:, 0, 0])
    else:
        markers = data['processed_X']
        ground_truth = data['processed_ground_truth'] if data['processed_ground_truth'] is not None else None

    # If the model can return the member predictions, do so.
    logging.info(f'Imputing markers: {pass_direction} pass')
    save_path_direction = None
    if save_path is not None:
        save_path_direction = os.path.join(save_path, f'{os.path.basename(data_file)}_{pass_direction}')
        if not os.path.exists(save_path_direction):
            os.mkdir(save_path_direction)

    preds, bad_frames, member_stds = predict_markers(model, dict_training, markers, bad_frames,
                                                            dataset_constants.KEYPOINTS,
                                                     dataset_constants.DIVIDER,
                                                     ground_truth,
                                                            # markers_to_fix=markers_to_fix,
                                                            # error_diff_thresh=error_diff_thresh,
                                                            save_path=save_path_direction,
                                                            exclude_value=exclude_value)


    # Flip the data for the reverse cases to save in the correct direction.
    if pass_direction == 'reverse':
        # markers = markers[:, ::-1, :]
        # ground_truth = ground_truth[:, ::-1, :] if ground_truth is not None else None
        preds = preds[:, ::-1, :]
        bad_frames = bad_frames[:, ::-1, :]
        member_stds = member_stds[:, ::-1, :]
        print(preds.shape, member_stds.shape, bad_frames.shape)


    # Save predictions to a matlab file.
    if save_path is not None:
        file_name = f'{os.path.basename(data_file).split(".csv")[0]}_{pass_direction}.mat'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, file_name)
        logging.info(f'Saving to {save_path}')
        output_dict = {'preds': preds,
                        'markers': data['X'], # inputs to the model, z-scored
                       'marker_names': dataset_constants.KEYPOINTS,
                        'bad_frames': bad_frames, # original to keep track of where were the holes
                       'data_file': data_file,
                        'pass_direction': pass_direction,
                        'marker_means': transform_dict["marker_means"],
                        'marker_stds': transform_dict["marker_stds"],
                        'rot_angle': transform_dict["rot_angle"],
                        'mean_position': transform_dict["mean_position"],
                        'exclude_value': exclude_value,
                        'ground_truth': data['ground_truth'] if data['ground_truth'] is not None else [],
                        'member_stds': member_stds}
        logging.debug(f'{output_dict}')
        savemat(save_path, output_dict)

    return preds

