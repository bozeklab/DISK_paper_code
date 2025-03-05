"""
Inspiration from CAPTURE paper and CAPTURE_demo matlab scripts
https://github.com/jessedmarshall/CAPTURE_demo/blob/b85581c796237634c50f715549519a9b98507867/Utility/align_hands_elbows.m
"""

import numpy as np
from scipy.signal import medfilt
from utils import get_mask
import logging

def get_ref_point(X, marker_names, point, divider):
    if type(point) == list:
        if len(point) > 1:
            point_index = [marker_names.index(kp) for kp in point]
            return np.mean([X[:, :, p * divider: p * divider + divider] for p in point_index], axis=0)
        else:
            point_index = marker_names.index(point[0])
            return X[:, :, point_index * divider: point_index * divider + divider]
    elif type(point) == str:
        point_index = marker_names.index(point)
        return X[:, :, point_index * divider: point_index * divider + divider]

    else:
        raise ValueError(f'point is expected to be either a list or a string, got type {type(point)} with value {point}')

def pad_vector(vect, n_pad):
    return np.array([vect[0], ] * n_pad + list(vect) + [vect[-1], ] * n_pad)


def z_score_data(input, exclude_value=-4668):
    """
    Z-score the marker data across time

    :param input: np array of shape (samples, time, keypoints * 3)
    :param exclude_value:

    :return:
    """
    input_with_nans = np.copy(input)
    input_with_nans[get_mask(input, exclude_value)] = np.nan

    marker_means = np.nanmean(input_with_nans, axis=1)
    marker_means = marker_means[:, np.newaxis]

    marker_stds = np.nanstd(input_with_nans, axis=1)
    marker_stds = marker_stds[:, np.newaxis]

    z_score_input = (input_with_nans - marker_means) / (marker_stds + 1e-9)
    z_score_input[np.isnan(z_score_input)] = exclude_value

    return z_score_input, marker_means, marker_stds

def apply_z_score(input, marker_means, marker_stds, exclude_value=-4668):
    """
    Z-score the marker data across time

    :param input: np array of shape (samples, time, keypoints * 3)
    :param exclude_value:

    :return:
    """
    input_with_nans = np.copy(input)
    input_with_nans[get_mask(input, exclude_value)] = np.nan

    z_score_input = (input_with_nans - marker_means) / (marker_stds + 1e-9)
    z_score_input[np.isnan(z_score_input)] = exclude_value

    return z_score_input


def preprocess_data(X, marker_names, divider, front_point, middle_point, exclude_value):
    """
    Preprocess frame by frame for every sample

    :param X: np.array of shape (samples, time, markers * 3D)
    :param marker_names: list of strings, marker names

    :return:
    """
    # 1. medfilt, kernel=3
    # X = medfilt(X, kernel_size=3)
    orig_X = np.copy(X)
    filt_X = np.copy(X)
    orig_X[get_mask(orig_X, exclude_value)] = np.nan
    for i in range(X.shape[0]):
        for j in range(X.shape[2]):
            x = medfilt(pad_vector(orig_X[i, :, j], 3), kernel_size=3)
            filt_X[i, :, j] = x[3: -3]

    # 2. egocentric frame transformation
    ## 2.1 get the marker rotation matrix
    X_front_point = get_ref_point(filt_X, marker_names, front_point, divider)
    X_middle_point = get_ref_point(filt_X, marker_names, middle_point, divider)

    # rot_angle = np.arctan2( - (X_front_point[..., 1] - X_middle_point[..., 1]),
    #                           (X_front_point[..., 0] - X_middle_point[..., 0]))
    if divider == 3:
        first_axis = np.array([1, 0, 0])
    else:
        first_axis = np.array([1, 0])
    vectx = X_front_point[..., 0] - X_middle_point[..., 0], X_front_point[..., 1] - X_middle_point[..., 1]
    rot_angle = np.arctan2(first_axis[1], first_axis[0]) - np.arctan2(vectx[1], vectx[0])

    ## 2.2 subtract the mean and rotate
    ego_X = filt_X.reshape(filt_X.shape[0], filt_X.shape[1], len(marker_names), divider) - X_middle_point[:, :, np.newaxis]
    rot_X = np.copy(ego_X)

    # modify the x coordinates
    rot_X[..., 0] = ego_X[..., 0] * np.cos(rot_angle[..., np.newaxis]) - ego_X[..., 1] * np.sin(rot_angle[..., np.newaxis])
    # modify the y coordinates
    rot_X[..., 1] = ego_X[..., 0] * np.sin(rot_angle[..., np.newaxis]) + ego_X[..., 1] * np.cos(rot_angle[..., np.newaxis])
    rot_X = rot_X.reshape(X.shape)

    rot_X[np.isnan(rot_X)] = exclude_value

    return rot_X, rot_angle, X_middle_point

def apply_transform(X, divider, rot_angle, mean_position, marker_means, marker_stds, exclude_value):
    X[get_mask(X, exclude_value)] = np.nan
    filt_X = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[2]):
            x = medfilt(pad_vector(X[i, :, j], 3), kernel_size=3)
            filt_X[i, :, j] = x[3: -3]

    # 2. egocentric frame transformation
    ## 2.2 subtract the mean and rotate
    ego_X = filt_X.reshape(filt_X.shape[0], filt_X.shape[1], -1, divider) - mean_position[:, :, np.newaxis]

    rot_X = np.copy(ego_X)
    # modify the x coordinates
    rot_X[..., 0] = ego_X[..., 0] * np.cos(rot_angle[..., np.newaxis]) - ego_X[..., 1] * np.sin(rot_angle[..., np.newaxis])
    # modify the y coordinates
    rot_X[..., 1] = ego_X[..., 0] * np.sin(rot_angle[..., np.newaxis]) + ego_X[..., 1] * np.cos(rot_angle[..., np.newaxis])

    z_score_input = (rot_X - marker_means) / (marker_stds + 1e-9)
    z_score_input[np.isnan(z_score_input)] = exclude_value

    return z_score_input


def fill_nan_forward(arr):
    mask = np.isnan(arr)
    if len(arr.shape) == 2:
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = arr[np.arange(idx.shape[0])[:, None], idx]
    elif len(arr.shape) == 3:
        arr2 = np.swapaxes(arr, 2, 1)
        new_arr = np.vstack([fill_nan_forward(arr2D)[np.newaxis, :] for arr2D in arr2])
        out = np.swapaxes(new_arr, 1, 2)
    else:
        raise ValueError(f'support only arr with 2 and 3 dimensions')
    return out

def unprocess_data(X, divider, rot_angle, mean_position, marker_means, marker_stds, marker_names, exclude_value):
    # undo the z-score
    logging.info(f'[IN UNPROCESS_DATA], markers: {np.unique(X)[0]} {np.unique(X)[-1]} {exclude_value}')

    if np.any(get_mask(marker_means, exclude_value)):
        # marker_means[get_mask(marker_means, exclude_value)] = np.nan
        marker_means = fill_nan_forward(marker_means)

    if np.any(get_mask(marker_stds, exclude_value)):
        # marker_stds[get_mask(marker_stds, exclude_value)] = np.nan
        marker_stds = fill_nan_forward(marker_stds)

    if np.any(get_mask(rot_angle, exclude_value)):
        # rot_angle[get_mask(rot_angle, exclude_value)] = np.nan
        rot_angle = fill_nan_forward(rot_angle)

    if np.any(get_mask(mean_position, exclude_value)):
        # mean_position[get_mask(mean_position, exclude_value)] = np.nan
        mean_position = fill_nan_forward(mean_position)

    # undo the z-scoring
    unz_X = X * marker_stds + marker_means
    logging.info(f'[IN UNPROCESS_DATA], un_Z: {np.unique(unz_X)[0]} {np.unique(unz_X)[-1]} {exclude_value}')
    unz_X = unz_X.reshape(X.shape[0], X.shape[1], len(marker_names), divider)
    # unz_X = np.copy(X).reshape(X.shape[0], X.shape[1], len(marker_names), 3)

    # undo the rotation
    unrot_X = np.copy(unz_X)
    unrot_X[..., 0] = unz_X[..., 0] * np.cos(rot_angle[..., np.newaxis]) + unz_X[..., 1] * np.sin(rot_angle[..., np.newaxis])
    unrot_X[..., 1] = - unz_X[..., 0] * np.sin(rot_angle[..., np.newaxis]) + unz_X[..., 1] * np.cos(rot_angle[..., np.newaxis])
    logging.info(f'[IN UNPROCESS_DATA], unrot_X: {np.unique(unrot_X)[0]} {np.unique(unrot_X)[-1]} {exclude_value}')

    # undo the centering
    unproc_X = (unrot_X + mean_position[:, :, np.newaxis]).reshape(X.shape)
    logging.info(f'[IN UNPROCESS_DATA], unproc_X: {np.unique(unproc_X)[0]} {np.unique(unproc_X)[-1]} {exclude_value}')
    unproc_X[get_mask(X, exclude_value)] = exclude_value
    logging.info(f'[IN UNPROCESS_DATA], final: {np.unique(unproc_X)[0]} {np.unique(unproc_X)[-1]} {exclude_value}')

    return unproc_X

