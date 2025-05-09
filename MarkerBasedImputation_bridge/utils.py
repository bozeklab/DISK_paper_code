"""Utitlity functions for mbi."""
import h5py
import numpy as np
import os
import shutil
import torch


def get_mask(X, value):
    """

    :param X:
    :type X:
    :param value:
    :type value:
    :return: boolean array
    """
    if value is np.nan:
        return np.isnan(X)
    else:
        return X == value

def load_dataset(data_path):
    """Load marker, marker_means, marker_stds, and bad_frames from .h5 dataset.

    :param data_path: Path to .h5 file
    Outputs:
    markers - Z-scored marker trajectories over time.
    marker_means - Mean of markers in real world coordinates
    marker_stds - Std of markers in real world coordinates
    bad_frames - Logical or integer matrix denoting frames in which there were
                 marker errors. Same size as markers.
                 0 when marker j is a good recording at frame i, 1 otherwise.
    """
    f = h5py.File(data_path, 'r')
    markers = np.array(f['markers'][:]).T
    marker_means = np.array(f['marker_means'][:]).T
    marker_stds = np.array(f['marker_stds'][:]).T
    bad_frames = np.array(f['bad_frames'][:]).T
    move_frames = np.array(f['move_frames'][:]).T

    return markers, marker_means, marker_stds, bad_frames, move_frames


def get_ids(bad_frames, input_length, output_length, only_good_inputs=False,
            only_good_outputs=False):
    """Get the sequences of indices to use for network training.

    Inputs:
    bad_frames - Logical or integer matrix denoting frames in which there were
                 marker errors. Same size as markers. 0 when marker j is a good
                 recording at frame i, 1 otherwise.
    input_length - Number of frames to input to the model.
    output_length - Number of frames the model will attempt to predict.

    Optional:
    only_good_inputs - Only return samples in which there are no bad frames in
                       the input sequence. Default: False
    only_good_outputs - Only return samples in which there are no bad frames in
                        the output sequence. Default: False

    Outputs:
    input_ids - N x input_length integer matrix of input ids.
    output_ids - N x output_length integer matrix of output ids.
    """
    # Find all the good frames
    n_bad_markers = np.sum(bad_frames, 1)
    good_frames = np.where(n_bad_markers == 0)[0]
    good_frames = \
        good_frames[(good_frames > input_length) &
                    (good_frames < (bad_frames.shape[0]-output_length))]

    # Save the preceding input_length ids before that frame
    input_ids = np.zeros([good_frames.shape[0], input_length], 'int32')
    output_ids = np.zeros([good_frames.shape[0], output_length], 'int32')

    for i in range(input_ids.shape[0]):
        input_ids[i, :] = range(good_frames[i]-input_length, good_frames[i])
        output_ids[i, :] = range(good_frames[i], good_frames[i]+output_length)

    # Remove all samples that have bad frames in the input
    if only_good_inputs:
        n_bad_markers_all = n_bad_markers[input_ids]
        is_bad_input = n_bad_markers_all != 0
        is_bad_sample = np.any(is_bad_input, axis=1)
        input_ids = input_ids[~is_bad_sample, :]
        output_ids = output_ids[~is_bad_sample, :]

    # Remove all samples that have bad frames in the output
    if only_good_outputs:
        n_bad_markers_all = n_bad_markers[output_ids]
        is_bad_output = n_bad_markers_all != 0
        is_bad_sample = np.any(is_bad_output, axis=1)
        input_ids = input_ids[~is_bad_sample, :]
        output_ids = output_ids[~is_bad_sample, :]

    return input_ids, output_ids


def create_run_folders(run_name, base_path="models", clean=False):
    """Create subfolders necessary for outputs of training.

    :param run_name: Name of the model folder
    :param base_path: Base path in which to store models
    :param clean: If True, deletes the contents of the run output path
    """
    def is_empty_run(run_path):
        weights_path = os.path.join(run_path, "weights")
        has_weights_folder = os.path.exists(weights_path)
        return not has_weights_folder or len(os.listdir(weights_path)) == 0

    run_path = os.path.join(base_path, run_name)

    if not clean:
        initial_run_path = run_path
        i = 1
        while os.path.exists(run_path):
            run_path = "%s_%02d" % (initial_run_path, i)
            i += 1

    if os.path.exists(run_path):
        shutil.rmtree(run_path)

    os.makedirs(run_path)
    print("Created folder:", run_path)

    return run_path


def asymmetric_temporal_padding(x, left_pad=1, right_pad=1):
    """Pad the middle dimension of a 3D tensor.

    with "left_pad" zeros left and "right_pad" right.
    """
    pattern = [[0, 0], [left_pad, right_pad], [0, 0]]
    return torch.pad(x, pattern)


def categorical_mean_squared_error(y_true, y_pred):
    """MSE for categorical variables."""
    return torch.mean(torch.square(torch.argmax(y_true, axis=-1) -
                      torch.argmax(y_pred, axis=-1)))
