"""Imputes markers with mbi models."""
# import clize
import datetime
import h5py
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import savemat, loadmat
from skimage import measure
import json
import logging
from glob import glob
import matplotlib
if os.uname().nodename == 'france-XPS':
    matplotlib.use('TkAgg')
    basedir = '/home/france/Mounted_dir'
else:
    matplotlib.use('Agg')
    basedir = '/projects/ag-bozek/france'

from preprocess_data import unprocess_data
from utils import get_mask


def sigmoid(x, x_0, k):
    """Sigmoid function.

    For use in weighted averaging of marker predictions from
    the forward and reverse passes.
    :param x: domain
    :param x_0: midpoint
    :parak k: exponent constant.
    """
    return 1 / (1 + np.exp(-k * (x - x_0)))


def merge(save_path, fold_paths):
    """Merge the predictions from chunked passes.

    :param save_path: Path to .mat file where merged predictions will be saved.
    :param fold_paths: List of paths to chunked predictions to merge.
    """
    # Order the files in the imputation path by the fold id in the filename
    # Consider revising
    # fold_files = [os.path.basename(s) for s in fold_paths]
    # folds = [int(re.findall('\d+', s)[0]) for s in fold_files]
    # sorted_indices = sorted(range(len(folds)), key=lambda k: folds[k])
    # fold_paths = [fold_paths[i] for i in sorted_indices]
    # print('Reorganized fold paths:')
    # print(fold_paths)

    n_folds_to_merge = len(fold_paths)
    markers = None
    bad_framesF = None
    bad_framesR = None
    predsF = None
    predsR = None

    member_stdsF = None
    member_stdsR = None
    for i in range(n_folds_to_merge):
        # print('%d' % i, flush=True)

        data = loadmat(fold_paths[i])
        pass_direction = data['pass_direction'][:]
        markers_single_fold = np.array(data['markers'][:])
        preds_single_fold = np.array(data['preds'][:])
        member_stds_single_fold = np.array(data['member_stds'][:])
        bad_frames_single_fold = np.array(data['bad_frames'][:])

        if (markers is None) & (pass_direction == 'forward'):
            markers = markers_single_fold
        elif pass_direction == 'forward':
            markers = np.concatenate((markers, markers_single_fold), axis=1)

        if (bad_framesF is None) & (pass_direction == 'forward'):
            bad_framesF = bad_frames_single_fold
        elif pass_direction == 'forward':
            bad_framesF = np.concatenate((bad_framesF, bad_frames_single_fold), axis=1)

        if (bad_framesR is None) & (pass_direction == 'reverse'):
            bad_framesR = bad_frames_single_fold
        elif pass_direction == 'reverse':
            bad_framesR = np.concatenate((bad_framesR, bad_frames_single_fold), axis=1)

        if (predsF is None) & (pass_direction == 'forward'):
            predsF = preds_single_fold
        elif pass_direction == 'forward':
            predsF = np.concatenate((predsF, preds_single_fold), axis=1)

        if (predsR is None) & (pass_direction == 'reverse'):
            predsR = preds_single_fold
        elif pass_direction == 'reverse':
            predsR =  np.concatenate((predsR, preds_single_fold), axis=1)

        if (member_stdsF is None) & (pass_direction == 'forward'):
            member_stdsF = member_stds_single_fold
        elif pass_direction == 'forward':
            member_stdsF = np.concatenate((member_stdsF, member_stds_single_fold), axis=1)

        if (member_stdsR is None) & (pass_direction == 'reverse'):
            member_stdsR = member_stds_single_fold
        elif pass_direction == 'reverse':
            member_stdsR = np.concatenate((member_stdsR, member_stds_single_fold), axis=1)

    marker_means = np.array(data['marker_means'][:])
    marker_stds = np.array(data['marker_stds'][:])
    rot_angle = np.array(data['rot_angle'][:])
    mean_position = np.array(data['mean_position'][:])
    marker_names = np.array(data['marker_names'][:])
    original_data_file = data['data_file'][0]
    exclude_value = data['exclude_value'][0][0]
    del data

    logging.info(f'shape markers: {markers.shape}')
    logging.info(f'shape member_stdsF: {member_stdsF.shape}')
    logging.info(f'shape predsF: {predsF.shape}')
    logging.info(f'shape bad_framesF: {bad_framesF.shape}')
    logging.info(f'shape markers_means: {marker_means.shape}')
    logging.info(f'shape marker_stds: {marker_stds.shape}')
    # Convert to real world coordinates
    # for i in range(markers.shape[1]):
    #     # looping on the time
    #     markers[:, i] = markers[:, i] * marker_stds[:, 0] + marker_means[:, 0]
    #     predsF[:, i] = predsF[:, i] * marker_stds[:, 0] + marker_means[:, 0]
    #     predsR[:, i] = predsR[:, i] * marker_stds[:, 0] + marker_means[:, 0]

    items = np.random.choice(predsF.shape[0], 10)
    for item in items:
        fig, axes = plt.subplots(predsF.shape[-1]//3, 3, figsize=(10, 10), sharey='col')
        axes = axes.flatten()
        for i in range(predsF.shape[-1]):
            x = markers[item, :, i]
            x[get_mask(x, exclude_value)] = np.nan
            t = np.arange(markers.shape[1])
            axes[i].plot(x, 'o-')
            axes[i].plot(t[bad_framesF[item, :, i].astype(bool)], predsF[item, bad_framesF[item, :, i].astype(bool), i], 'x')
            if i%3 == 0:
                axes[i].set_ylabel(marker_names[i//3])
        plt.savefig(os.path.join(save_path, f'single_predF_pred_item-{item}.png'))
        plt.close()

    markers = unprocess_data(markers, rot_angle, mean_position, marker_means, marker_stds, marker_names, exclude_value)
    predsF = unprocess_data(predsF, rot_angle, mean_position, marker_means, marker_stds, marker_names, exclude_value)
    predsR = unprocess_data(predsR, rot_angle, mean_position, marker_means, marker_stds, marker_names, exclude_value)

    for item in items:
        fig, axes = plt.subplots(predsF.shape[-1]//3, 3, figsize=(10, 10), sharey='col')
        axes = axes.flatten()
        for i in range(predsF.shape[-1]):
            x = markers[item, :, i]
            x[get_mask(x, exclude_value)] = np.nan
            t = np.arange(markers.shape[1])
            axes[i].plot(x, 'o-')
            axes[i].plot(t[bad_framesF[item, :, i].astype(bool)], predsF[item, bad_framesF[item, :, i].astype(bool), i], 'x')
            if i%3 == 0:
                axes[i].set_ylabel(marker_names[i//3])
        plt.savefig(os.path.join(save_path, f'single_predF_pred_item-{item}_after_unprocess.png'))
        plt.close()

    # This is not necessarily all the error frames from
    # multiple_predict_recording_with_replacement, but if they overlap,
    # we would just take the weighted average.
    bad_frames = np.zeros((bad_framesF.shape[0], bad_framesF.shape[1], np.round(bad_framesF.shape[2] / 3).astype('int32')))
    # 3 because 3D?
    for i in range(bad_frames.shape[2]):
        bad_frames[..., i] = np.any(bad_framesF[..., i * 3: i * 3 + 3] & bad_framesR[..., i * 3: i * 3 + 3], axis=2)

    # Compute the weighted average of the forward and reverse predictions using
    # a logistic function
    logging.info('Computing weighted average')
    preds = np.zeros(predsF.shape)
    member_stds = np.zeros(member_stdsF.shape)
    k = 1 # sigmoid exponent
    start = datetime.datetime.now()
    for sample in range(len(predsF)):
        # for i in range(bad_frames.shape[2] * 3): # *3
        is_bad = bad_frames[sample] #bad_frames[..., np.floor(i / 3).astype('int32')]
        for kp in range(bad_frames.shape[-1]):
            CC = measure.label(is_bad[:, kp], background=0)
            num_CC = len(np.unique(CC)) - 1
            # initialize to forward prediction
            preds[sample, ..., 3 * kp: 3 * kp + 3] = predsF[sample, ..., 3 * kp: 3 * kp +3]
            for j in range(num_CC):
                time_ids = np.where(CC == j + 1)[0]
                length_CC = len(time_ids)
                x_0 = np.round(length_CC / 2)
                weightR = sigmoid(np.arange(length_CC), x_0, k)[:, np.newaxis]
                weightF = 1 - weightR
                preds[sample, time_ids, kp * 3: kp * 3 + 3] = predsF[sample, time_ids, kp * 3: kp * 3 + 3] * weightF + predsR[sample, time_ids, kp * 3: kp * 3 + 3] * weightR
                member_stds[sample, time_ids, kp * 3: kp * 3 + 3] = np.sqrt(member_stdsF[sample, time_ids, kp * 3: kp * 3 + 3]**2 * weightF + member_stdsR[sample, time_ids, kp * 3: kp * 3 + 3]**2 * weightR)
    elapsed = datetime.datetime.now() - start
    logging.info(f'Computing average took: {elapsed} seconds')

    # Save predictions to a matlab file.
    if save_path is not None:
        logging.info(f'Saving to {save_path}')
        with h5py.File(os.path.join(save_path, f'{os.path.basename(original_data_file).split(".csv")[0]}_final_predictions.h5'), "w") as f:
            f.create_dataset("preds", data=preds) # merged predictions
            f.create_dataset("markers", data=markers) # input to the ensemble models, de-z-scored
            f.create_dataset("badFrames", data=bad_frames)
            f.create_dataset("predsF", data=predsF)
            f.create_dataset("predsR", data=predsR)
            f.create_dataset("member_stds", data=member_stds) # should be 0 where no prediction, else a float giving the divergence of the ensemble models (std)

        # save it in the same csv format as the other methods, so it is easier to compare
        cols = [f'{i//3}_{i%3 + 1}' for i in range(preds.shape[2])]
        for i_sample in range(preds.shape[0]):
            output_file_path = os.path.join(save_path, f'{os.path.basename(original_data_file).split(".csv")[0]}_sample{i_sample}_MBI.csv')
            df = pd.DataFrame(columns=cols, data = preds[i_sample])
            df['behaviour'] = np.nan
            df.to_csv(output_file_path, index=False)
    return preds

if __name__ == "__main__":
    # Wrapper for running from commandline
    save_path = os.path.join(basedir, 'results_behavior/MarkerBasedImputation/model_ensemble_01/')
    fold_paths = glob(os.path.join(basedir, 'results_behavior/MarkerBasedImputation/model_ensemble_01/test_repeat-0*.mat'))

    merge(save_path, fold_paths)