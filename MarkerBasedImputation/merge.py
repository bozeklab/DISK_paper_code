"""Imputes markers with mbi models."""
# import clize
import datetime
import h5py
import numpy as np
import os
import re

import pandas as pd
from scipy.io import savemat, loadmat
from skimage import measure
import json
from glob import glob
import matplotlib
if os.uname().nodename == 'france-XPS':
    matplotlib.use('TkAgg')
    basedir = '/home/france/Mounted_dir'
else:
    matplotlib.use('Agg')
    basedir = '/projects/ag-bozek/france'


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
    print(fold_paths)

    n_folds_to_merge = len(fold_paths)
    markers = None
    bad_framesF = None
    bad_framesR = None
    predsF = None
    predsR = None

    member_stdsF = None
    member_stdsR = None
    for i in range(n_folds_to_merge):
        print('%d' % i, flush=True)

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
    data = None

    print(markers.shape)
    print(member_stdsF.shape)
    print(predsF.shape)
    print(bad_framesF.shape)
    print(marker_means.shape)
    print(marker_stds.shape, flush=True)
    # Convert to real world coordinates
    # for i in range(markers.shape[1]):
    #     # looping on the time
    #     markers[:, i] = markers[:, i] * marker_stds[:, 0] + marker_means[:, 0]
    #     predsF[:, i] = predsF[:, i] * marker_stds[:, 0] + marker_means[:, 0]
    #     predsR[:, i] = predsR[:, i] * marker_stds[:, 0] + marker_means[:, 0]
    markers = markers * marker_stds + marker_means
    predsF = predsF * marker_stds + marker_means
    predsR = predsR * marker_stds + marker_means

    # This is not necessarily all the error frames from
    # multiple_predict_recording_with_replacement, but if they overlap,
    # we would just take the weighted average.
    bad_frames = np.zeros((bad_framesF.shape[0], bad_framesF.shape[1], np.round(bad_framesF.shape[2] / 3).astype('int32')))
    # 3 because 3D?
    for i in range(bad_frames.shape[2]):
        bad_frames[..., i] = np.any(bad_framesF[..., i * 3: i * 3 + 3] & bad_framesR[..., i * 3: i * 3 + 3], axis=2)

    # Compute the weighted average of the forward and reverse predictions using
    # a logistic function
    print('Computing weighted average:', flush=True, end=' ')
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
    print(elapsed)

    # Save predictions to a matlab file.
    if save_path is not None:
        s = 'Saving to %s' % (save_path)
        print(s)
        with h5py.File(os.path.join(save_path, 'final_predictions.h5'), "w") as f:
            f.create_dataset("preds", data=preds) # merged predictions
            f.create_dataset("markers", data=markers) # input to the ensemble models, de-z-scored
            f.create_dataset("badFrames", data=bad_frames)
            f.create_dataset("predsF", data=predsF)
            f.create_dataset("predsR", data=predsR)
            f.create_dataset("member_stds", data=member_stds) # should be 0 where no prediction, else a float giving the divergence of the ensemble models (std)

        # save it in the same csv format as the other methods, so it is easier to compare
        cols = [f'{i//3}_{i%3}' for i in range(preds.shape[2])]
        for i_sample in range(preds.shape[0]):
            output_file_path = os.path.join(save_path, f'test_repeat-0_sample{i_sample}_MBI.csv')
            df = pd.DataFrame(columns=cols, data = preds[i_sample])
            df['behaviour'] = np.nan
            df.to_csv(output_file_path, index=False)
    return preds

if __name__ == "__main__":
    # Wrapper for running from commandline
    save_path = os.path.join(basedir, 'results_behavior/MarkerBasedImputation/model_ensemble_03_merged/')
    fold_paths = glob(os.path.join(basedir, 'results_behavior/MarkerBasedImputation/model_ensemble_03_preds/*.mat'))

    merge(save_path, fold_paths)