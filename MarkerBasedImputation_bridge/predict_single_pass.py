"""Imputes markers with mbi models."""
# import clize
import h5py
# from keras.models import load_model
import numpy as np
import os
from scipy.io import savemat
from scipy import stats
from glob import glob
import torch
import pandas as pd
import matplotlib
import json

from build_ensemble import EnsembleModel
from preprocess_data import z_score_data, preprocess_data
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

    if 'repeat' in filepath:
        df = pd.read_csv(filepath, sep='|')

        input = np.vstack([np.array(eval(v))[np.newaxis] for v in df['input']])
        input = input.reshape(input.shape[0], input.shape[1], -1)
        ground_truth = np.vstack([np.array(eval(v))[np.newaxis] for v in df['label']])
        ground_truth = ground_truth.reshape(input.shape[0], input.shape[1], -1)

        exclude_value = -4668
    else:
        df = pd.read_csv(filepath, sep=',')
        cols = df.columns

        input = df[cols].values[np.newaxis]
        ground_truth = np.copy(input)

        exclude_value = np.nan

    transformed_coords, rot_angle, mean_position = preprocess_data(input, dataset_constants.KEYPOINTS,
                                                                        middle_point=['right_hip', 'left_hip'],
                                                                        front_point=['right_coord', 'left_coord'])

    # Z-score the marker data
    z_score_input, marker_means, marker_stds = z_score_data(transformed_coords, exclude_value=exclude_value)

    transforms_dict = {'rot_angle': rot_angle,
                       'mean_position': mean_position,
                       'marker_means': marker_means,
                       'marker_stds': marker_stds,
                       'exclude_value': exclude_value}

    return z_score_input, ground_truth, dataset_constants, transforms_dict


def predict_markers(model, dict_model, X, bad_frames, markers_to_fix=None,
                    error_diff_thresh=.25, outlier_thresh=3, device=torch.device('cpu')):
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
    # bad_frames = np.repeat(bad_frames, 3, axis=1) > .5

    # See whether you should fix errors
    fix_errors = np.any(markers_to_fix)

    # Reshape and get the starting seed.
    # X = X[None, ...]
    bad_frames_any = np.any(bad_frames, axis=2)
    startpoint = np.argmax(bad_frames_any, axis=1)
    startpoint = np.clip(startpoint - input_length - 1, a_min=-input_length-1, a_max=X.shape[1] - 1)
    next_frame_id = startpoint + input_length + 1
    mask = next_frame_id < X.shape[1] - output_length
    # Preallocate
    preds = np.copy(X)
    pred = np.zeros((X.shape[0], output_length, X.shape[2]))

    member_stds = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    member_pred = np.zeros((X.shape[0], model.n_members, X.shape[2]))

    # At each step, generate a prediction, replace the predictions of
    # markers you do not want to predict with the ground truth, and append
    # the resulting vector to the end of the next input chunk.
    while np.max(startpoint[mask]) > -input_length-1:

        X_start = np.vstack([x[np.array([max(0, t) for t in range(s, s + input_length)])][np.newaxis] for (x, s) in zip(preds[mask], startpoint[mask])])
        # If there is a marker prediction that is greater than the
        # difference threshold above, mark it as a bad frame.
        # These are likely just jumps or identity swaps from MoCap that
        # were not picked up by preprocessing.
        if fix_errors:
            diff = pred[:, 0, :] - X[:, next_frame_id, :]
            errors = np.squeeze(np.abs(diff) > error_diff_thresh)
            errors[~markers_to_fix] = False
            bad_frames[next_frame_id, errors] = True
        if np.any(bad_frames[next_frame_id, :]):
            pred, member_pred = model(torch.Tensor(X_start).to(device))
            pred = pred.detach().cpu().numpy()
            member_pred = member_pred.detach().cpu().numpy()

        # Detect anomalous predictions.
        outliers = np.squeeze(np.abs(pred) > outlier_thresh)

        pred[:, 0][outliers] = preds[mask, next_frame_id[mask]][outliers]

        # Only use the predictions for the bad markers. Take the
        # predictions and append to the end of X_start for future
        # prediction.
        pred[:, 0][~bad_frames[mask, next_frame_id[mask]]] = preds[mask, next_frame_id[mask]][~bad_frames[mask, next_frame_id[mask]]]
        # print(pred[0, 0], X[mask][0, next_frame_id[mask][0]])
        for i_member in range(0, model.n_members):
            member_pred[:, i_member, 0][~bad_frames[mask, next_frame_id[mask]]] = float('nan')
        member_std = np.nanstd(member_pred, axis=1)
        if np.any(np.isnan(member_std)):
            print('NANs!')
        preds[mask, next_frame_id[mask]] = np.squeeze(pred)
        member_stds[mask, next_frame_id[mask], :] = np.squeeze(member_std)

        bad_frames = preds == -4668
        bad_frames_any = np.any(bad_frames, axis=2)
        print(np.sum(bad_frames_any), end =' ')
        startpoint = np.argmax(bad_frames_any, axis=1)
        startpoint = np.clip(startpoint - input_length - 1, -input_length-1, X.shape[1] - 1)
        next_frame_id = startpoint + input_length + 1
        mask = next_frame_id < X.shape[1] - output_length

    return preds, bad_frames, member_stds



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
    markers, ground_truth, dataset_constants, transform_dict = open_data_csv(filepath=data_file, dataset_path=dataset_path)
    bad_frames = markers == -transform_dict['exclude_value'] # value used by optipose to mark dropped frames
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
        print('Loading model')
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
        bad_frames = bad_frames[:, ::-1, :]

    # print('Predicting %d frames starting at frame %d.'
    #       % (markers.shape[1], start_frame))

    # If the model can return the member predictions, do so.
    print('Imputing markers: %s pass' % pass_direction, flush=True)
    preds, _, member_stds = predict_markers(model, dict_training, markers, bad_frames,
                                            markers_to_fix=markers_to_fix,
                                            error_diff_thresh=error_diff_thresh)


    # Flip the data for the reverse cases to save in the correct direction.
    if pass_direction == 'reverse':
        markers = markers[:, ::-1, :]
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
        print('Saving to %s' % save_path)
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