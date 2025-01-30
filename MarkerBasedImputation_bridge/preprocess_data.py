"""
Inspiration from CAPTURE paper and CAPTURE_demo matlab scripts
https://github.com/jessedmarshall/CAPTURE_demo/blob/b85581c796237634c50f715549519a9b98507867/Utility/align_hands_elbows.m
"""

import os, sys
import tqdm
from glob import glob

import numpy as np
import pandas as pd
from skimage.io import imread, imsave
import math
import matplotlib.pyplot as plt
from scipy.signal import medfilt

def get_ref_point(X, marker_names, point):
    if type(point) == list:
        if len(point) > 1:
            point_index = [marker_names.index(kp) for kp in point]
            return np.mean([X[:, :, p * 3: p * 3 + 3] for p in point_index], axis=0)
        else:
            point_index = marker_names.index(point[0])
            return X[:, :, point_index * 3: point_index * 3 + 3]
    elif type(point) == str:
        point_index = marker_names.index(point)
        return X[:, :, point_index * 3: point_index * 3 + 3]

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
    input_with_nans[input == exclude_value] = np.nan

    marker_means = np.nanmean(input_with_nans, axis=1)
    marker_means = marker_means[:, np.newaxis]

    marker_stds = np.nanstd(input_with_nans, axis=1)
    marker_stds = marker_stds[:, np.newaxis]

    z_score_input = (input_with_nans - marker_means) / (marker_stds + 1e-9)
    z_score_input[np.isnan(z_score_input)] = exclude_value

    return z_score_input, marker_means, marker_stds


def preprocess_data(X, marker_names, front_point, middle_point):
    """
    Preprocess frame by frame for every sample

    :param X: np.array of shape (samples, time, markers * 3D)
    :param marker_names: list of strings, marker names

    :return:
    """
    # 1. medfilt, kernel=3
    # X = medfilt(X, kernel_size=3)
    filt_X = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[2]):
            x = medfilt(pad_vector(X[i, :, j], 3), kernel_size=3)
            filt_X[i, :, j] = x[3: -3]

    # 2. egocentric frame transformation
    ## 2.1 get the marker rotation matrix
    X_front_point = get_ref_point(filt_X, marker_names, front_point)
    X_middle_point = get_ref_point(filt_X, marker_names, middle_point)

    # rot_angle = np.arctan2( - (X_front_point[..., 1] - X_middle_point[..., 1]),
    #                           (X_front_point[..., 0] - X_middle_point[..., 0]))
    first_axis = np.array([1, 0, 0])
    vectx = X_front_point[..., 0] - X_middle_point[..., 0], X_front_point[..., 1] - X_middle_point[..., 1]
    rot_angle = np.arctan2(first_axis[1], first_axis[0]) - np.arctan2(vectx[1], vectx[0])

    ## 2.2 subtract the mean and rotate
    ego_X = filt_X.reshape(filt_X.shape[0], filt_X.shape[1], len(marker_names), 3) - X_middle_point[:, :, np.newaxis]
    rot_X = np.copy(ego_X)

    # modify the x coordinates
    rot_X[..., 0] = ego_X[..., 0] * np.cos(rot_angle[..., np.newaxis]) - ego_X[..., 1] * np.sin(rot_angle[..., np.newaxis])
    # modify the y coordinates
    rot_X[..., 1] = ego_X[..., 0] * np.sin(rot_angle[..., np.newaxis]) + ego_X[..., 1] * np.cos(rot_angle[..., np.newaxis])

    return ego_X.reshape(X.shape), rot_angle, X_middle_point

def unprocess_data(X, rot_angle, mean_position, marker_means, marker_stds, marker_names):
    # undo the z-score
    unz_X = X * marker_stds + marker_means
    unz_X = unz_X.reshape(X.shape[0], X.shape[1], len(marker_names), 3)

    # undo the rotation
    unrot_X = np.copy(unz_X)
    unrot_X[..., 0] = unz_X[..., 0] * np.cos(rot_angle[..., np.newaxis]) + unz_X[..., 1] * np.sin(rot_angle[..., np.newaxis])
    unrot_X[..., 1] = - unz_X[..., 0] * np.sin(rot_angle[..., np.newaxis]) + unz_X[..., 1] * np.cos(rot_angle[..., np.newaxis])

    # undo the centering
    return (unrot_X + mean_position[:, :, np.newaxis]).reshape(X.shape)

