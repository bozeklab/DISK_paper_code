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
from DISK.utils.utils import read_constant_file

if os.uname().nodename == 'france-XPS':
    matplotlib.use('TkAgg')
    basedir = '/home/france/Mounted_dir'
else:
    matplotlib.use('Agg')
    basedir = '/projects/ag-bozek/france'


def open_data_csv(filepath, dataset_path, stride=1):
    """Load data from csv file used for comparison."""

    dataset_constant_file = glob(os.path.join(dataset_path, 'constants.py'))[0]
    dataset_constants = read_constant_file(dataset_constant_file)

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
        ground_truth = np.copy(input)
        logging.info(f'input shape: {input.shape}')
        exclude_value = np.nan

    return input, ground_truth, dataset_constants, exclude_value
    # processed_GT, rot_angle, mean_position, marker_means, marker_stds = preprocess_data(ground_truth, dataset_constants.KEYPOINTS, ## TEST TO UNDO
    #                                                                     middle_point=['right_hip', 'left_hip'],
    #                                                                     front_point=['right_coord', 'left_coord'],
    #                                                                exclude_value=exclude_value)
    #
    # transformed_coords = apply_transform(input, rot_angle, mean_position, marker_means, marker_stds,  exclude_value)
    #
    # # unproc_X = unprocess_data(transformed_coords, rot_angle, mean_position, marker_means, marker_stds, dataset_constants.KEYPOINTS, exclude_value)
    # #
    # # items = np.random.choice(transformed_coords.shape[0], 5)
    # # for item in items:
    # #     fig, axes = plt.subplots(transformed_coords.shape[-1]//3, 3, figsize=(10, 10))
    # #     axes = axes.flatten()
    # #     for i in range(transformed_coords.shape[-1]):
    # #         x = ground_truth[item, :, i]
    # #         x[x == exclude_value] = np.nan
    # #         axes[i].plot(x, 'o-')
    # #
    # #         x = input[item, :, i]
    # #         x[x == exclude_value] = np.nan
    # #         axes[i].plot(x, 'x')
    # #
    # #     fig, axes = plt.subplots(transformed_coords.shape[-1]//3, 3, figsize=(10, 10), sharey='col')
    # #     axes = axes.flatten()
    # #     for i in range(transformed_coords.shape[-1]):
    # #         x = processed_GT[item, :, i]
    # #         x[x == exclude_value] = np.nan
    # #         axes[i].plot(x, 'o-')
    # #
    # #         x = transformed_coords[item, :, i]
    # #         x[x == exclude_value] = np.nan
    # #         axes[i].plot(x, 'x')
    #
    # transforms_dict = {'rot_angle': rot_angle,
    #                    'mean_position': mean_position,
    #                    'marker_means': marker_means,
    #                    'marker_stds': marker_stds,
    #                    'exclude_value': exclude_value}
    #
    # return transformed_coords, processed_GT, dataset_constants, transforms_dict


def predict_markers(model, dict_model, X, bad_frames, keypoints, ground_truth=None, markers_to_fix=None,
                    error_diff_thresh=.25, outlier_thresh=3, device=torch.device('cpu'), save_path='',
                    exclude_value=-4668, pass_direction='forward'):
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



    processed_ground_truth, rot_angle_GT, mean_position_GT = preprocess_data(ground_truth,
                                                                             keypoints,
                                                                             middle_point=['right_hip', 'left_hip'],
                                                                             front_point=['right_coord', 'left_coord'],
                                                                             exclude_value=exclude_value)

    processed_X, rot_angle, mean_position = preprocess_data(X,
                                                                  keypoints,
                                                                  middle_point=['right_hip', 'left_hip'],
                                                                  front_point=['right_coord', 'left_coord'],
                                                                  exclude_value=exclude_value)

    _, marker_means, marker_stds = z_score_data(processed_X.reshape(1, -1, processed_X.shape[2]), exclude_value=exclude_value)

    processed_ground_truth = apply_z_score(processed_ground_truth, marker_means, marker_stds, exclude_value=exclude_value)

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

        processed_X_start, rot_angle, mean_position = preprocess_data(X_start,
                                                                keypoints,
                                                                middle_point=['right_hip', 'left_hip'],
                                                                front_point=['right_coord', 'left_coord'],
                                                                exclude_value=exclude_value)
        processed_X_start = apply_z_score(processed_X_start, marker_means, marker_stds, exclude_value=-4668)

        # If there is a marker prediction that is greater than the
        # difference threshold above, mark it as a bad frame.
        # These are likely just jumps or identity swaps from MoCap that
        # were not picked up by preprocessing.
        if fix_errors:
            diff = pred[:, 0, :] - X[:, next_frame_id, :]
            errors = np.squeeze(np.abs(diff) > error_diff_thresh)
            errors[~markers_to_fix] = False
            bad_frames[next_frame_id, errors] = True

        pred, member_pred = model(torch.Tensor(processed_X_start).to(device))
        pred = pred.detach().cpu().numpy()
        member_pred = member_pred.detach().cpu().numpy()
        # Detect anomalous predictions.
        # outliers = np.squeeze(np.abs(pred) > outlier_thresh)
        # pred[:, 0][outliers] = preds[mask, next_frame_id[mask]][outliers]
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
                    x[x == exclude_value] = np.nan
                    axes[i].plot(list(t) + [9], processed_ground_truth[mask][item][next_frame_id[mask][item]-9:next_frame_id[mask][item] + 1][:, i], 'o-', label='GT')
                    axes[i].plot(x, 'o-', label='input')
                    if bad_frames[mask, next_frame_id[mask]][item, i]:
                        axes[i].plot(9, pred[item, 0, i], 'x', c= 'red', label='pred wo missing data')
                    else:
                        axes[i].plot(9, pred[item, 0, i], 'x', c='cyan', label='pred w missing data')
                    if i%3 == 0:
                        axes[i].set_ylabel(keypoints[i//3])
                    if i==2:
                        axes[i].legend()
                plt.suptitle(f'RMSE {rmse[item]:.3f}')
                plt.savefig(os.path.join(save_path, f'single_pred_single_step_{pass_direction}_item-{item}.png'))
                plt.close()
            print('stop')


        # Only use the predictions for the bad markers. Take the
        # predictions and append to the end of X_start for future
        # prediction.
        pred[:, 0][~bad_frames[mask, next_frame_id[mask]]] = preds[mask, next_frame_id[mask]][~bad_frames[mask, next_frame_id[mask]]]
        for i_member in range(0, model.n_members):
            member_pred[:, i_member, 0][~bad_frames[mask, next_frame_id[mask]]] = float('nan')
        member_std = np.nanstd(member_pred, axis=1)

        if np.all(np.isnan(member_std)):
            logging.info(f'NANs: {bad_frames[mask, next_frame_id[mask]][::3].shape}, {np.any(bad_frames[mask, next_frame_id[mask]][::3], axis=-1)}')

        preds[mask, next_frame_id[mask]] = np.squeeze(pred)
        member_stds[mask, next_frame_id[mask], :] = np.squeeze(member_std)

        bad_frames = preds == exclude_value
        bad_frames_any = np.any(bad_frames, axis=2)  # axis=2 is the keypoint axis
        logging.info(f'Progress: remaining missing values = {np.sum(bad_frames_any)}')
        startpoint = np.argmax(bad_frames_any, axis=1)  # returns the first point of missing = next_frame_id
        next_frame_id = startpoint
        startpoint = np.clip(startpoint - input_length, a_min=-input_length - 1, a_max=X.shape[1] - 1)
        mask = next_frame_id > 0

    bad_frames_orig = processed_X == exclude_value
    for item in np.random.choice(preds.shape[0], 10):
        fig, axes = plt.subplots(pred.shape[-1]//3, 3, figsize=(10, 10), sharey='col', sharex='all')
        axes = axes.flatten()
        for i in range(pred.shape[-1]):
            t = np.arange(X.shape[1])
            if ground_truth is not None:
                axes[i].plot(t, processed_ground_truth[item, :, i], 'o-')
            else:
                x = processed_X[item, :, i]
                x[x == exclude_value] = np.nan
                axes[i].plot(x, 'o-')
            axes[i].plot(t[bad_frames_orig[item, :, i]], preds[item, bad_frames_orig[item, :, i], i], 'x')
        plt.savefig(os.path.join(save_path, f'single_pred_{pass_direction}_item-{item}.png'))
        plt.close()

    transforms_dict = {'rot_angle': rot_angle,
                       'mean_position': mean_position,
                       'marker_means': marker_means,
                       'marker_stds': marker_stds}

    return preds, bad_frames, member_stds, transforms_dict



def predict_single_pass(model_path, data_file, dataset_path, pass_direction, *,
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
    if not (pass_direction == 'forward') | (pass_direction == 'reverse'):
        raise ValueError('pass_direction must be forward or reverse')

    # # Check data extensions
    # filename, file_extension = os.path.splitext(data_path)
    # accepted_extensions = {'.h5', '.hdf5', '.mat'}
    # if file_extension not in accepted_extensions:
    #     raise ValueError('Improper extension: hdf5 or \
    #                      mat -v7.3 file required.')
    #
    # # Load data
    # print('Loading data')
    # f = h5py.File(data_path, 'r')
    # if file_extension in {'.h5', '.hdf5'}:
    #     markers = np.array(f['markers'][:]).T
    #     marker_means = np.array(f['marker_means'][:]).T
    #     marker_stds = np.array(f['marker_stds'][:]).T
    #     bad_frames = np.array(f['bad_frames'][:]).T
    # else:
    #     # Get the markers data from the struct
    #     dset = 'markers_aligned_preproc'
    #     marker_names = list(f[dset].keys())
    #     n_frames_tot = f[dset][marker_names[0]][:].T.shape[0]
    #     n_dims = f[dset][marker_names[0]][:].T.shape[1]
    #
    #     markers = np.zeros((n_frames_tot, len(marker_names)*n_dims))
    #     for i in range(len(marker_names)):
    #         marker = f[dset][marker_names[i]][:].T
    #         for j in range(n_dims):
    #             markers[:, i*n_dims + j] = marker[:, j]
    #
    #     # Z-score the marker data
    #     marker_means = np.mean(markers, axis=0)
    #     marker_means = marker_means[None, ...]
    #     marker_stds = np.std(markers, axis=0)
    #     marker_stds = marker_stds[None, ...]
    #     print(marker_means)
    #     print(marker_stds)
    #     markers = stats.zscore(markers)
    #
    #     # Get the bad_frames data from the cell
    #     dset = 'bad_frames_agg'
    #     n_markers = f[dset][:].shape[0]
    #     bad_frames = np.zeros((markers.shape[0], n_markers))
    #     for i in range(n_markers):
    #         reference = f[dset][i][0]
    #         bad_frames[np.squeeze(f[reference][:]).astype('int32') - 1, i] = 1
    #
    # # Get the start frame and number of frames after splitting the data up
    # markers = markers[::stride, :]
    # bad_frames = bad_frames[::stride, :]
    # n_frames = int(np.floor(markers.shape[0]/n_folds))
    # markers, ground_truth, dataset_constants, transform_dict = open_data_csv(filepath=data_file, dataset_path=dataset_path)
    # bad_frames = markers == transform_dict['exclude_value'] # value used by optipose to mark dropped frames

    markers, ground_truth, dataset_constants, exclude_value = open_data_csv(filepath=data_file, dataset_path=dataset_path)
    bad_frames = markers == exclude_value # value used by optipose to mark dropped frames

    # fold_id = int(fold_id)
    # start_frame = markers.shape[1] * int(fold_id)
    # n_frames = int(np.floor(markers.shape[1] / n_folds))

    # # Also predict the remainder if on the last fold.
    # if fold_id == (n_folds-1):
    #     markers = markers[:, start_frame:, :]
    #     bad_frames = bad_frames[:, start_frame:, :]
    # else:
    #     markers = markers[:, start_frame: start_frame + n_frames, :]
    #     bad_frames = bad_frames[:, start_frame: start_frame + n_frames, :]

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
        markers = markers[:, ::-1, :]
        ground_truth = ground_truth[:, ::-1, :]
        bad_frames = bad_frames[:, ::-1, :]

    # print('Predicting %d frames starting at frame %d.'
    #       % (markers.shape[1], start_frame))

    # If the model can return the member predictions, do so.
    logging.info(f'Imputing markers: {pass_direction} pass')
    preds, _, member_stds, transform_dict = predict_markers(model, dict_training, markers, bad_frames,
                                                            dataset_constants.KEYPOINTS, ground_truth,
                                            markers_to_fix=markers_to_fix,
                                            error_diff_thresh=error_diff_thresh,
                                            save_path=save_path,
                                            exclude_value=exclude_value,
                                                            pass_direction=pass_direction)


    # Flip the data for the reverse cases to save in the correct direction.
    if pass_direction == 'reverse':
        markers = markers[:, ::-1, :]
        ground_truth = ground_truth[:, ::-1, :]
        preds = preds[:, ::-1, :]
        bad_frames = bad_frames[:, ::-1, :]
        member_stds = member_stds[:, ::-1, :]

    # Save predictions to a matlab file.
    if save_path is not None:
        # file_name = '%s_fold_id_%d.mat' % (pass_direction, fold_id)
        file_name = f'{os.path.basename(data_file).split(".csv")[0]}_{pass_direction}.mat'
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
                        'pass_direction': pass_direction,
                        'marker_means': transform_dict["marker_means"],
                        'marker_stds': transform_dict["marker_stds"],
                        'rot_angle': transform_dict["rot_angle"],
                        'mean_position': transform_dict["mean_position"],
                        'exclude_value': exclude_value,
                        'ground_truth': ground_truth,
                        'member_stds': member_stds}
        savemat(save_path, output_dict)

    return preds

if __name__ == '__main__':
    model_ensemble_path = os.path.join(basedir, 'results_behavior/MarkerBasedImputation/model_ensemble_01/final_model.h5')
    dataset_path = os.path.join(basedir, 'results_behavior/datasets/INH_FL2_keypoints_1_60_wresiduals_w1nan_stride0.5_new')
    # data_file = os.path.join(basedir, 'results_behavior/outputs/25-09-24_FL2_new_for_comparison/DISK_test/test_for_optipose_repeat_0/test_w-all-nans_file0.csv')
    data_file = os.path.join(basedir, 'results_behavior/models/test_CLB_optipose_debug/test_for_optipose_repeat_0/test_repeat-0.csv')
    save_path = os.path.join(basedir, 'results_behavior/MarkerBasedImputation/model_ensemble_01')

    impute_stride = 5
    nfolds = 1 # 20
    # folds are only for faster imputation, drop it
    errordiff_th = 0.5

    for i_fold in range(nfolds):
        for pass_direction in ['reverse', 'forward']:
            predict_single_pass(model_ensemble_path, data_file, dataset_path, pass_direction,
                                save_path=save_path, stride=impute_stride, n_folds=nfolds, fold_id=i_fold,
                                markers_to_fix=None, error_diff_thresh=errordiff_th,
                                model=None)