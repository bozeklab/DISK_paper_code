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


# def simple_median_filter(X, k=3, axis=1):
#     X_shape = X.shape
#     idx = np.delete(np.arange(len(X_shape)), axis)
#     other_dims = X_shape[idx]
#     if len(other_dims) == 1:
#         I = [slice(None)] * X.ndim
#         I[other_dims[0]] = slice(0, X.shape[other_dims[0]])
#         [pad_vector(X[i], k - 2) for i in I]
#
#     elif len(other_dims) == 2:
#
#     elif len(other_dims) == 3:
#
#     else:
#         return ValueError(f'simple_median_filter not implemented for array with more than {len(other_dims) + 1}')

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
    X_front_point = get_ref_point(X, marker_names, front_point)
    X_middle_point = get_ref_point(X, marker_names, middle_point)

    rot_angle = np.arctan2( - (X_front_point[..., 1] - X_middle_point[..., 1]),
                              (X_front_point[..., 0] - X_middle_point[..., 0]))
    first_axis = np.array([1, 0, 0])
    vectx = X_front_point[..., 0] - X_middle_point[..., 0], X_front_point[..., 1] - X_middle_point[..., 1]
    rot_angle = np.arctan2(first_axis[1], first_axis[0]) - np.arctan2(vectx[1], vectx[0])

    ## 2.2 subtract the mean and rotate
    ego_X = X.reshape(X.shape[0], X.shape[1], len(marker_names), 3) - X_middle_point[:, :, np.newaxis]
    rot_X = np.copy(ego_X)

    # modify the x coordinates
    rot_X[..., 0] = ego_X[..., 0] * np.cos(rot_angle[..., np.newaxis]) - ego_X[..., 1] * np.sin(rot_angle[..., np.newaxis])
    # modify the y coordinates
    rot_X[..., 1] = ego_X[..., 0] * np.sin(rot_angle[..., np.newaxis]) + ego_X[..., 1] * np.cos(rot_angle[..., np.newaxis])

    return ego_X.reshape(X.shape), rot_angle, X_middle_point

