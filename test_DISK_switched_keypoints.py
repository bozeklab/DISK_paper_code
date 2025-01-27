import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba
import argparse
import os, sys
import re
from tqdm import tqdm
import importlib.util
import pandas as pd
from sklearn.decomposition import PCA
from glob import glob
from omegaconf import OmegaConf
import logging
from sklearn.cluster import KMeans, AffinityPropagation
import seaborn as sns
import plotly.express as px
import gc
from scipy.spatial.transform import Rotation
from matplotlib import gridspec

from DISK.utils.utils import read_constant_file, load_checkpoint
from DISK.utils.dataset_utils import load_datasets
from DISK.utils.transforms import init_transforms
from DISK.utils.train_fillmissing import construct_NN_model
from DISK.models.graph import Graph

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

dict_label2int = {'INH1A': {'experiment': {'INH1A': 0},
                            'treatment': {'A': 0, 'B': 1, 'C': 2},
                            'treatment_detail': {'A': 0, 'B': 1.5, 'C': 2.5},
                            'mouse_id': {'M1': 0, 'M2': 1, 'M3': 2, 'M4': 3, 'M5': 4,
                                         'M6': 5, 'M7': 6, 'M8': 7, 'M9': 8, 'M10': 9},
                            'experiment_type': {'FL2': 0, 'CLB': 1}
                            },
                  'INH2B': {'experiment': {'INH2B': 1},
                            'treatment': {'A': 0, 'B': 3, 'C': 1, 'D': 2, 'E': 4, 'F': 5},
                            'treatment_detail': {'A': 0, 'B': 3, 'C': 1, 'D': 2, 'E': 4, 'F': 5},
                            'mouse_id': {'M1': 10, 'M2': 11, 'M3': 12, 'M4': 13, 'M5': 14, 'M6': 15,
                                         'M7': 16, 'M8': 17, 'M9': 18, 'M10': 19, 'M11': 20, 'M12': 21},
                            'experiment_type': {'FL2': 0, 'CLB': 1}
                            },
                  'CP1': {'experiment': {'CP1': 2},
                          'treatment': {'A': 0, 'B': 6, 'C': 6, 'D': 6},
                          'treatment_detail': {'A': 0, 'B': 6, 'C': 6.3, 'D': 6.6},
                          'mouse_id': {'M1': 22, 'M14': 23, 'M15': 24, 'M19': 25},
                          'experiment_type': {'FL2': 0, 'CLB': 1}
                          },
                  'CP1B': {'experiment': {'CP1B': 3},
                           'treatment': {'A': 0, 'B': 6, 'C': 6, 'D': 6},
                           'treatment_detail': {'A': 0, 'B': 6, 'C': 6.3, 'D': 6.6},
                           'mouse_id': {'M1': 26, 'M2': 27, 'M3': 28, 'M4': 29, 'M5': 30, 'M6': 31},
                           'experiment_type': {'FL2': 0, 'CLB': 1}
                           },
                  'MOS1aD': {'experiment': {'MOS1aD': 4},
                             'treatment': {'A': 0, 'B': 6, 'C': 1, 'D': 2, 'E': 7},
                             'treatment_detail': {'A': 0, 'B': 6.6, 'C': 1.5, 'D': 2.5, 'E': 7},
                             'mouse_id': {'M4': 32, 'M5': 33, 'M6': 34, 'M7': 35, 'M8': 36, 'M9': 37, 'M10': 38},
                             'experiment_type': {'CLB': 1},
                             }
                  }

reverse_dict_treatment_detail = {0.0: 'vehicle',
                                 1.0: 'PF3845 10mg/kg',
                                 1.5: 'PF3945 30mg/kg',
                                 2.0: 'MJN110 1.25mg/kg',
                                 2.5: 'MJN110 2.5mg/kg',
                                 3.0: 'AM251 3mg/kg',
                                 4.0: 'AM251 + PF3845',
                                 5.0: 'AM251 + MJN110',
                                 6.0: 'CP55,940 0.03',
                                 6.3: 'CP55,940 0.01',
                                 6.6: 'CP55,940 0.3',
                                 7.0: 'Harmaline 20 mg/kg'}

reverse_dict_experiment = {0: 'INH1A', 1: 'INH2B', 2: 'CP1A', 3: 'CP1B', 4: 'MOS1aD'}

### In create_bogna_dataset, saved multi class indices in this order:
label_order4 = ['experiment', 'treatment', 'mouse_id', 'experiment_type']  # for retro active compatibility
label_order = ['experiment', 'treatment', 'treatment_detail', 'mouse_id', 'experiment_type']
reverse_dict_experiment_type = {0: 'FL2', 1: 'CLB'}

def statistics_mouse(input_tensor, dataset_constants, device):
    coordinates = input_tensor[:, :, :, :dataset_constants.DIVIDER]

    barycenter = torch.mean(coordinates[:, :, :6, :], dim=2)

    movement = torch.mean(torch.abs(torch.diff(coordinates, dim=1)), dim=(1, 2, 3))
    speed_z = torch.mean(torch.diff(barycenter[..., 2], dim=1), dim=1)
    speed_xy = torch.mean(torch.diff(barycenter[..., :2], dim=1), dim=(1, 2))
    upside_down = torch.mean(torch.mean(coordinates[:, :, 6:, 2]) - barycenter[..., 2], dim=1)
    ## average height
    average_height = torch.mean(coordinates[:, :, :, 2], dim=(1, 2))
    ## back length
    mean_hips = torch.mean(coordinates[:, :, :2, :], dim=2)
    mean_shoulders = torch.mean(coordinates[:, :, 4:6, :], dim=2)
    back_length = torch.sqrt(torch.sum((mean_hips - mean_shoulders) ** 2, dim=-1))  # of shape (batch, time)
    back_length = torch.mean(back_length, dim=1)  # shape (batch,)
    ## relative mobility of shoulders

    ### distance barycenter-shoulder
    dist_barycenter_shoulders = torch.sqrt(
        torch.sum((barycenter - mean_shoulders) ** 2, dim=-1))  # of shape (batch, time)
    dist_barycenter_shoulders = torch.mean(dist_barycenter_shoulders, dim=1)
    ### relative height of shoulder wrt the back
    height_shoulders = torch.mean(mean_shoulders[:, :, 2] - barycenter[:, :, 2],
                                  dim=-1)  ## look only at z coordinates
    ### angle of shoulders wrt the back
    mean_hips_coords = torch.mean(coordinates[:, :, :4, :], dim=2)
    barycenter_back_vect = (mean_hips - mean_hips_coords)[:, :, :2]
    barycenter_shoulder_vect = (mean_shoulders - mean_hips_coords)[:, :, :2]
    # atan2(v2.y,v2.x) - atan2(v1.y,v1.x)
    angleXY_shoulders = torch.mean(
        torch.atan2(barycenter_shoulder_vect[:, :, 0], barycenter_shoulder_vect[:, :, 1]) - torch.atan2(
            barycenter_back_vect[:, :, 0], barycenter_back_vect[:, :, 1]), dim=1)
    diff_angleXY_shoulders = torch.atan2(barycenter_shoulder_vect[:, -1, 0], barycenter_shoulder_vect[:, -1, 1]) - torch.atan2(
            barycenter_back_vect[:, 0, 0], barycenter_back_vect[:, 0, 1])
    angleXY_shoulders_base = torch.mean(
        torch.atan2(barycenter_shoulder_vect[:, :, 0], barycenter_shoulder_vect[:, :, 1]) - torch.atan2(
            torch.zeros_like(barycenter_back_vect[:, :, 0]), torch.ones_like(barycenter_back_vect[:, :, 0])).to(
            device), dim=1)

    # distance between knees
    dist_bw_knees = torch.mean(
        torch.sqrt(torch.sum((coordinates[:, :, 6, :] - coordinates[:, :, 7, :]) ** 2, dim=-1)), dim=1)
    # distance between shoulders and knees
    mean_knees = torch.mean(coordinates[:, :, 6:, :], dim=2)
    dist_knees_shoulders = torch.mean(torch.sqrt(torch.sum((mean_shoulders - mean_knees) ** 2, dim=-1)), dim=1)

    return (movement, upside_down, speed_xy, speed_z, average_height, back_length, dist_barycenter_shoulders,
            height_shoulders, angleXY_shoulders, diff_angleXY_shoulders, dist_bw_knees, dist_knees_shoulders,
            angleXY_shoulders_base)


def statistics_human(input_tensor, dataset_constants, device):
    coordinates = input_tensor[:, :, :, :dataset_constants.DIVIDER]

    barycenter = torch.mean(coordinates[:, :, np.array([0, 12, 16]), :], dim=2)

    movement = torch.mean(torch.abs(torch.diff(coordinates, dim=1)), dim=(1, 2, 3))
    speed_z = torch.mean(torch.diff(barycenter[..., 2], dim=1), dim=1)
    speed_xy = torch.mean(torch.diff(barycenter[..., :2], dim=1), dim=(1, 2))
    upside_down = torch.mean(torch.mean(coordinates[:, :, np.array([16, 17, 18, 19, 12, 13, 14, 15]), 2], dim=1)
                             - torch.mean(coordinates[:, :, np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]), 2]), dim=1)
    ## average height
    average_height = torch.mean(coordinates[:, :, :, 2], dim=(1, 2))
    ## back length
    mean_hips = torch.mean(coordinates[:, :, np.array([0, 12, 16]), :], dim=2)
    mean_shoulders = torch.mean(coordinates[:, :, np.array([2, 3, 4, 8]), :], dim=2)
    back_length = torch.sqrt(torch.sum((mean_hips - mean_shoulders) ** 2, dim=-1))  # of shape (batch, time)
    back_length = torch.mean(back_length, dim=1)  # shape (batch,)
    ## relative mobility of shoulders

    ### distance barycenter-shoulder
    dist_barycenter_shoulders = torch.sqrt(
        torch.sum((barycenter - mean_shoulders) ** 2, dim=-1))  # of shape (batch, time)
    dist_barycenter_shoulders = torch.mean(dist_barycenter_shoulders, dim=1)
    ### relative height of shoulder wrt the back
    height_shoulders = torch.mean(mean_shoulders[:, :, 2] - barycenter[:, :, 2],
                                  dim=-1)  ## look only at z coordinates
    ### angle of shoulders wrt the back
    mean_hips_coords = torch.mean(coordinates[:, :, np.array([0, 1, 12, 16]), :], dim=2)
    barycenter_back_vect = (mean_hips - mean_hips_coords)[:, :, :2]
    barycenter_shoulder_vect = (mean_shoulders - mean_hips_coords)[:, :, :2]
    # atan2(v2.y,v2.x) - atan2(v1.y,v1.x)
    angleXY_shoulders = torch.mean(
        torch.atan2(barycenter_shoulder_vect[:, :, 0], barycenter_shoulder_vect[:, :, 1]) - torch.atan2(
            barycenter_back_vect[:, :, 0], barycenter_back_vect[:, :, 1]), dim=1)
    angleXY_shoulders_base = torch.mean(
        torch.atan2(barycenter_shoulder_vect[:, :, 0], barycenter_shoulder_vect[:, :, 1]) - torch.atan2(
            torch.zeros_like(barycenter_back_vect[:, :, 0]), torch.ones_like(barycenter_back_vect[:, :, 0])).to(
            device), dim=1)

    # distance between knees
    dist_bw_knees = torch.mean(
        torch.sqrt(torch.sum((coordinates[:, :, 13, :] - coordinates[:, :, 17, :]) ** 2, dim=-1)), dim=1)
    # distance between shoulders and knees
    mean_knees = torch.mean(coordinates[:, :, np.array([13, 17]), :], dim=2)
    dist_knees_shoulders = torch.mean(torch.sqrt(torch.sum((mean_shoulders - mean_knees) ** 2, dim=-1)), dim=1)

    return (movement, upside_down, speed_xy, speed_z, average_height, back_length, dist_barycenter_shoulders,
            height_shoulders, angleXY_shoulders, dist_bw_knees, dist_knees_shoulders, angleXY_shoulders_base)

def statistics_MABe(input_tensor, dataset_constants, device):
    """
    Careful true keypoints are the following, not the ones
    KEYPOINTS = ['animal0_snout', 'animal0_leftear', 'animal0_rightear', 'animal0_neck', 'animal0_left', 'animal0_right', 'animal0_tail',
                 'animal1_snout', 'animal1_leftear', 'animal1_rightear', 'animal1_neck', 'animal1_left', 'animal1_right', 'animal1_tail']
    """
    coordinates = input_tensor[:, :, :, :dataset_constants.DIVIDER]

    barycenter = torch.mean(coordinates[:, :, :, :], dim=2)

    movement = torch.mean(torch.abs(torch.diff(coordinates, dim=1)), dim=(1, 2, 3))
    movement_mouse1 = torch.mean(torch.abs(torch.diff(coordinates[:, :, :7], dim=1)), dim=(1, 2, 3))
    movement_mouse2 = torch.mean(torch.abs(torch.diff(coordinates[:, :, 7:], dim=1)), dim=(1, 2, 3))
    movement_mouse1_mouse2 = torch.mean(torch.abs(
        torch.diff(coordinates[:, :, :7], dim=1) - torch.diff(coordinates[:, :, 7:],
                                                                               dim=1)), dim=(1, 2, 3))
    speed_xy = torch.mean(torch.diff(barycenter[..., :2], dim=1), dim=(1, 2))
    ## average height
    dist_bw_mice = torch.mean(torch.sum((torch.mean(coordinates[:, :, 3:7, :], dim=2) - torch.mean(
        coordinates[:, :, 11:, :], dim=2)) ** 2, dim=2), dim=1)
    angle_base = torch.mean(torch.atan2(barycenter[:, :, 0], barycenter[:, :, 1]) - torch.atan2(
        torch.zeros_like(barycenter[:, :, 0]), torch.ones_like(barycenter[:, :, 0])).to(device), dim=1)
    angle_2mice = torch.mean(
        torch.atan2(coordinates[:, :, 6, 0] - coordinates[:, :, 0, 0],
                    coordinates[:, :, 6, 1] - coordinates[:, :, 0, 1]) - torch.atan2(
            coordinates[:, :, 13, 0] - coordinates[:, :, 7, 0], coordinates[:, :, 13, 1] - coordinates[:, :, 7, 1]),
        dim=1)
    angle_mouse1 = torch.mean(
        torch.atan2(coordinates[:, :, 6, 0] - coordinates[:, :, 0, 0],
                    coordinates[:, :, 6, 1] - coordinates[:, :, 0, 1]) - torch.atan2(
        torch.zeros_like(barycenter[:, :, 0]), torch.ones_like(barycenter[:, :, 0])).to(device),
        dim=1)
    angle_mouse2 = torch.mean(
        torch.atan2(coordinates[:, :, 13, 0] - coordinates[:, :, 7, 0],
                    coordinates[:, :, 13, 1] - coordinates[:, :, 7, 1]) - torch.atan2(
        torch.zeros_like(barycenter[:, :, 0]), torch.ones_like(barycenter[:, :, 0])).to(device),
        dim=1)

    return (movement, movement_mouse1, movement_mouse2, movement_mouse1_mouse2, speed_xy,
            dist_bw_mice, angle_base, angle_2mice, angle_mouse1, angle_mouse2)


def statistics_fish(input_tensor, dataset_constants):
    coordinates = input_tensor[:, :, :, :dataset_constants.DIVIDER]

    barycenter = torch.mean(coordinates[:, :, :, :], dim=2)

    movement = torch.mean(torch.abs(torch.diff(coordinates, dim=1)), dim=(1, 2, 3))
    movement_fish1 = torch.mean(torch.abs(torch.diff(coordinates[:, :, np.array([0, 1, 2])], dim=1)), dim=(1, 2, 3))
    movement_fish2 = torch.mean(torch.abs(torch.diff(coordinates[:, :, np.array([3, 4, 5])], dim=1)), dim=(1, 2, 3))
    movement_fish1_fish2 = torch.mean(torch.abs(torch.diff(coordinates[:, :, np.array([0, 1, 2])], dim=1) - torch.diff(coordinates[:, :, np.array([3, 4, 5])], dim=1)), dim=(1, 2, 3))
    speed_z = torch.mean(torch.diff(barycenter[..., 2], dim=1), dim=1)
    speed_xy = torch.mean(torch.diff(barycenter[..., :2], dim=1), dim=(1, 2))
    ## average height
    average_height = torch.mean(coordinates[:, :, :, 2], dim=(1, 2))
    dist_bw_fishes = torch.mean(torch.sum((torch.mean(coordinates[:, :, np.array([0, 1, 2]), :], dim=2) - torch.mean(coordinates[:, :, np.array([3, 4, 5]), :], dim=2)) ** 2, dim=2), dim=1)
    angle_base = torch.mean(torch.atan2(barycenter[:, :, 0], barycenter[:, :, 1]) - torch.atan2(
            torch.zeros_like(barycenter[:, :, 0]), torch.ones_like(barycenter[:, :, 0])).to(device), dim=1)
    angle_2fish = torch.mean(
        torch.atan2(coordinates[:, :, 2, 0] - coordinates[:, :, 0, 0], coordinates[:, :, 2, 1] - coordinates[:, :, 0, 1]) - torch.atan2(
            coordinates[:, :, 5, 0] - coordinates[:, :, 3, 0], coordinates[:, :, 5, 1] - coordinates[:, :, 3, 1]), dim=1)

    return (movement, movement_fish1, movement_fish2, movement_fish1_fish2, speed_xy, speed_z, average_height,
            dist_bw_fishes, angle_base, angle_2fish)


def add_swap_keypoint_samples(data_dict, add_swap=True, proba=0.5):
    """
    For each sample in data_dict['X'], create a swap sample example with a proba 0.5
    Swapped keypoints procedure:
    1.decide with probability "proba" if creating a swap in the given sample
    2. choose 2 keypoints randomly (better would be to swap them with a probability relative to their distance in the dataset)
    3. choose randomly a time for the swap, uniform in the sample length at first.
    4. choose a start time for the swap, it needs to be at least partially outside the gap
    5. append the new sample to the data_dict

    Parameters
    ----------
    data_dict : data_dict output by the pytorch dataloader
    add_swap : boolean, True or False
    proba: probability for a given sample that we make a swap version of it

    Returns
    ----------

    """
    original_length = len(data_dict['X'])
    if not add_swap:
        swapped = [False] * original_length
        return data_dict, swapped

    new_xs = []
    new_x_supps = []
    new_indices_file = []
    new_indices_pos = []
    new_masks = []
    new_labels = []
    for x, x_supp, mask, i_file, i_pos, label in zip(data_dict['X'], data_dict['x_supp'], data_dict['mask_holes'],
                                                     data_dict['indices_file'], data_dict['indices_pos'],
                                                     data_dict['label']):
        # data_dict['X'] is a tensor on cpu
        # sample has shape (time, keypoints, dim)
        rd = np.random.random()
        if rd > proba:
            continue

        rd_kp = np.random.choice(a=x.shape[1], size=2, replace=False)
        rd_length = np.random.randint(1, x.shape[0])
        # mask1D = np.any(mask.detach().numpy(), axis=1) # False is non missing, True is missing
        # r0 = np.random.randint(0, np.argmax(mask1D) - 1) if np.argmax(mask1D) - 1 > 0 else 0
        # r1 = np.random.randint(1, np.argmax(mask1D[::-1]) + 1) if np.argmax(mask1D[::-1]) > 0 else 1
        # rd_start = np.random.choice([r0, len(mask1D) - r1])
        rd_start = np.random.choice(np.arange(0, x.shape[0] - rd_length))

        new_sample = x.detach().clone()
        new_sample[rd_start: rd_start + rd_length, rd_kp[0], :] = x[rd_start: rd_start + rd_length, rd_kp[1], :].detach().clone()
        new_sample[rd_start: rd_start + rd_length, rd_kp[1], :] = x[rd_start: rd_start + rd_length, rd_kp[0], :].detach().clone()
        # new_x_supp = x_supp.detach().clone()
        # new_x_supp[rd_start: rd_start + rd_length, rd_kp[0], :] = x_supp[rd_start: rd_start + rd_length, rd_kp[1], :]
        # new_x_supp[rd_start: rd_start + rd_length, rd_kp[1], :] = x_supp[rd_start: rd_start + rd_length, rd_kp[0], :]
        new_mask = mask.detach().clone()
        new_mask[rd_start: rd_start + rd_length, rd_kp[0]] = mask[rd_start: rd_start + rd_length, rd_kp[1]].detach().clone()
        new_mask[rd_start: rd_start + rd_length, rd_kp[1]] = mask[rd_start: rd_start + rd_length, rd_kp[0]].detach().clone()

        new_xs.append(new_sample)
        new_x_supps.append(x_supp.detach().clone())
        new_masks.append(new_mask)
        new_indices_file.append(i_file.detach().clone())
        new_indices_pos.append(i_pos.detach().clone())
        new_labels.append(label.detach().clone())

    swapped = np.array([False] * original_length + [True] * len(new_xs), dtype=bool)
    if len(new_xs) > 0:
        data_dict['X'] = torch.cat([data_dict['X'], torch.stack(new_xs, dim=0)])
        data_dict['x_supp'] = torch.cat([data_dict['x_supp'], torch.stack(new_xs, dim=0)])
        data_dict['mask_holes'] = torch.cat([data_dict['mask_holes'], torch.stack(new_masks, dim=0)])
        data_dict['indices_file'] = torch.cat([data_dict['indices_file'], torch.stack(new_indices_file, dim=0)])
        data_dict['indices_pos'] = torch.cat([data_dict['indices_pos'], torch.stack(new_indices_pos, dim=0)])
        data_dict['label'] = torch.cat([data_dict['label'], torch.stack(new_labels, dim=0)])
    return data_dict, swapped


def extract_hidden(model, data_loader, dataset_constants, model_cfg, device,
                   compute_statistics=False, pck_final_threshold=0.01, original_coordinates=False,
                   add_swap=True, proba=0.1,):
    label_ = []
    index_file = []
    index_pos = []
    hidden_array_ = []
    if len(dataset_constants.KEYPOINTS) == 14:
        statistics = {'movement': [],
                      'movement_mouse1': [],
                      'movement_mouse2': [],
                      'movement_mouse1-mouse2': [],
                      'speed_xy': [],
                      'dist_bw_mice': [],
                      'angle_2mice': [],
                      'angle_base': [],
                      'angle_mouse1': [],
                      'angle_mouse2': [],
                      'swapped': []
                     }
    elif len(dataset_constants.KEYPOINTS) > 10:
        statistics = {'movement': [],
                      'upside_down': [],
                      'speed_xy': [],
                      'speed_z': [],
                      'average_height': [],
                      'back_length': [],
                      'dist_barycenter_shoulders': [],
                      'height_shoulders': [],
                      'angleXY_shoulders': [],
                      'dist_bw_knees': [],
                      'dist_knees_shoulders': [],
                      'angle_back_base': [],
                      'swapped': []
                      }
    elif len(dataset_constants.KEYPOINTS) >= 8:
        statistics = {'movement': [],
                      'upside_down': [],
                      'speed_xy': [],
                      'speed_z': [],
                      'average_height': [],
                      'back_length': [],
                      'dist_barycenter_shoulders': [],
                      'height_shoulders': [],
                      'angleXY_shoulders': [],
                      'diff_angleXY_shoulders': [],
                      'dist_bw_knees': [],
                      'dist_knees_shoulders': [],
                      'angle_back_base': [],
                      'swapped': []
                      }
    else:
        statistics = {'movement': [],
                      'movement_fish1': [],
                      'movement_fish2': [],
                      'movement_fish1-fish2': [],
                      'speed_xy': [],
                      'speed_z': [],
                      'average_height': [],
                      'dist_bw_fishes': [],
                      'angle_2fish': [],
                      'angle_base': [],
                      'swapped': []
                     }
    statistics.update({'rmse': [], 'pck': [], 'mpjpe': [], 'n_missing': [], 'estimated_error': [],
                       'norm_estimated_error': []})


    for ith, data_dict in tqdm(enumerate(data_loader), total=len(data_loader), ascii=True, desc='Extract hidden'):
        data_dict, swapped = add_swap_keypoint_samples(data_dict, add_swap=add_swap, proba=proba)

        data_with_holes = data_dict['X'].to(device)
        mask_holes = data_dict['mask_holes'].to(device)

        input_tensor = data_dict['x_supp'].to(device)

        index_file.append(data_dict['indices_file'].detach().cpu().numpy())
        index_pos.append(data_dict['indices_pos'].detach().cpu().numpy())
        if 'label' in data_dict.keys():
            labels = data_dict['label']
            label_.append(torch.squeeze(labels, 1))
        if torch.sum(torch.isnan(input_tensor)) != 0:
            logging.warning('[extract_hidden] NANS in input tensor')

        input_tensor_with_holes = torch.cat([data_with_holes[..., :dataset_constants.DIVIDER], torch.unsqueeze(mask_holes, dim=-1)], dim=3)
        input_tensor_with_holes[:, 1:, :] = input_tensor_with_holes[:, :-1, :].clone()
        de_out = model.proj_input(input_tensor_with_holes, mask_holes)
        for i in range(model.num_layers):
            de_out = model.encoder_layers[i](de_out)

        ## metrics
        full_data_np = data_dict['x_supp'].detach().numpy()
        prediction = model(input_tensor_with_holes, mask_holes)
        prediction, uncertainty_estimate = prediction
        uncertainty_estimate = uncertainty_estimate.view(
            (input_tensor_with_holes.shape[0], input_tensor_with_holes.shape[1], input_tensor_with_holes.shape[2], -1))
        prediction = prediction.view(
            (input_tensor_with_holes.shape[0], input_tensor_with_holes.shape[1], input_tensor_with_holes.shape[2], -1))
        prediction_np = prediction.detach().cpu().numpy()
        mask_holes_np = data_dict['mask_holes'].detach().numpy()
        reshaped_mask_holes = np.repeat(mask_holes_np, dataset_constants.DIVIDER, axis=-1).reshape(full_data_np.shape)
        rmse = np.sqrt(np.mean(np.sum(((prediction_np - full_data_np) ** 2) * reshaped_mask_holes,
                                    axis=3), axis=(1,2)))  # sum on the XYZ dimension, output shape (batch, time, keypoint)
        euclidean_distance = np.sqrt(
            np.sum(((prediction_np - full_data_np) ** 2) * reshaped_mask_holes,
                   axis=3)) # sum on the XYZ dimension, output shape (batch, time, keypoint)
        mpjpe = np.mean(euclidean_distance, axis=(1,2))
        n_missing = np.sum(mask_holes_np, axis=(1, 2))
        pck = np.sum(euclidean_distance <= pck_final_threshold, axis=(1,2)) / n_missing
        estimated_error = np.sum(uncertainty_estimate.detach().cpu().numpy() * reshaped_mask_holes, axis=(1, 2, 3))

        if compute_statistics:  # TODO: implement with mask
            statistics['rmse'].extend(rmse)
            statistics['mpjpe'].extend(mpjpe)
            statistics['pck'].extend(pck)
            statistics['n_missing'].extend(n_missing)
            statistics['estimated_error'].extend(estimated_error)
            statistics['norm_estimated_error'].extend(estimated_error / n_missing)

            if len(dataset_constants.KEYPOINTS) == 14:
                (movement, movement_mouse1, movement_mouse2, movement_mouse1_mouse2, speed_xy,
                 dist_bw_mice, angle_base, angle_2mice, angle_mouse1, angle_mouse2) = statistics_MABe(input_tensor, dataset_constants, device)

                statistics['movement'].extend(movement.detach().cpu().numpy())
                statistics['movement_mouse1'].extend(movement_mouse1.detach().cpu().numpy())
                statistics['movement_mouse2'].extend(movement_mouse2.detach().cpu().numpy())
                statistics['movement_mouse1-mouse2'].extend(movement_mouse1_mouse2.detach().cpu().numpy())
                statistics['speed_xy'].extend(speed_xy.detach().cpu().numpy())
                statistics['dist_bw_mice'].extend(dist_bw_mice.detach().cpu().numpy())
                statistics['angle_2mice'].extend(angle_2mice.detach().cpu().numpy())
                statistics['angle_base'].extend(angle_base.detach().cpu().numpy())
                statistics['angle_mouse1'].extend(angle_mouse1.detach().cpu().numpy())
                statistics['angle_mouse2'].extend(angle_mouse2.detach().cpu().numpy())


            elif len(dataset_constants.KEYPOINTS) > 10:
                (movement, upside_down, speed_xy, speed_z, average_height, back_length, dist_barycenter_shoulders,
                 height_shoulders, angleXY_shoulders, dist_bw_knees, dist_knees_shoulders, angleXY_shoulders_base) = statistics_human(input_tensor, dataset_constants, device)

                statistics['movement'].extend(movement.detach().cpu().numpy())
                statistics['upside_down'].extend(upside_down.detach().cpu().numpy())
                statistics['speed_xy'].extend(speed_xy.detach().cpu().numpy())
                statistics['speed_z'].extend(speed_z.detach().cpu().numpy())
                statistics['average_height'].extend(average_height.detach().cpu().numpy())
                statistics['back_length'].extend(back_length.detach().cpu().numpy())
                statistics['dist_barycenter_shoulders'].extend(dist_barycenter_shoulders.detach().cpu().numpy())
                statistics['height_shoulders'].extend(height_shoulders.detach().cpu().numpy())
                statistics['angleXY_shoulders'].extend(angleXY_shoulders.detach().cpu().numpy())
                statistics['dist_bw_knees'].extend(dist_bw_knees.detach().cpu().numpy())
                statistics['dist_knees_shoulders'].extend(dist_knees_shoulders.detach().cpu().numpy())
                statistics['angle_back_base'].extend(angleXY_shoulders_base.detach().cpu().numpy())

            elif len(dataset_constants.KEYPOINTS) == 8:
                (movement, upside_down, speed_xy, speed_z, average_height, back_length, dist_barycenter_shoulders,
                 height_shoulders, angleXY_shoulders, diff_angleXY_shoulders, dist_bw_knees, dist_knees_shoulders,
                 angleXY_shoulders_base) = statistics_mouse(input_tensor, dataset_constants, device)

                statistics['movement'].extend(movement.detach().cpu().numpy())
                statistics['upside_down'].extend(upside_down.detach().cpu().numpy())
                statistics['speed_xy'].extend(speed_xy.detach().cpu().numpy())
                statistics['speed_z'].extend(speed_z.detach().cpu().numpy())
                statistics['average_height'].extend(average_height.detach().cpu().numpy())
                statistics['back_length'].extend(back_length.detach().cpu().numpy())
                statistics['dist_barycenter_shoulders'].extend(dist_barycenter_shoulders.detach().cpu().numpy())
                statistics['height_shoulders'].extend(height_shoulders.detach().cpu().numpy())
                statistics['angleXY_shoulders'].extend(angleXY_shoulders.detach().cpu().numpy())
                statistics['diff_angleXY_shoulders'].extend(diff_angleXY_shoulders.detach().cpu().numpy())
                statistics['dist_bw_knees'].extend(dist_bw_knees.detach().cpu().numpy())
                statistics['dist_knees_shoulders'].extend(dist_knees_shoulders.detach().cpu().numpy())
                statistics['angle_back_base'].extend(angleXY_shoulders_base.detach().cpu().numpy())

            elif len(dataset_constants.KEYPOINTS) == 6:
                (movement, movement_fish1, movement_fish2, movement_fish1_fish2, speed_xy, speed_z, average_height,
                dist_bw_fishes, angle_base, angle_2fish) = statistics_fish(input_tensor, dataset_constants)
                statistics['movement'].extend(movement.detach().cpu().numpy())
                statistics['movement_fish1'].extend(movement_fish1.detach().cpu().numpy())
                statistics['movement_fish2'].extend(movement_fish2.detach().cpu().numpy())
                statistics['movement_fish1-fish2'].extend(movement_fish1_fish2.detach().cpu().numpy())
                statistics['speed_xy'].extend(speed_xy.detach().cpu().numpy())
                statistics['speed_z'].extend(speed_z.detach().cpu().numpy())
                statistics['average_height'].extend(average_height.detach().cpu().numpy())
                statistics['dist_bw_fishes'].extend(dist_bw_fishes.detach().cpu().numpy())
                statistics['angle_2fish'].extend(angle_2fish.detach().cpu().numpy())
                statistics['angle_base'].extend(angle_base.detach().cpu().numpy())

        # statistics['swapped'].extend(data_dict['swap'].detach().cpu().numpy())
        statistics['swapped'].extend(swapped)
        hidden_array_.append(de_out.view(de_out.shape[0], de_out.shape[1] * de_out.shape[2]).detach().cpu().numpy())

    if len(label_) > 0:
        label_ = np.vstack(label_)
        logging.info(f'label_ shape:{label_.shape}')
    else:
        label_ = np.array([])
    hidden_array_ = np.vstack(hidden_array_)
    index_pos = np.concatenate(index_pos)
    index_file = np.concatenate(index_file)
    if compute_statistics:
        return hidden_array_, label_, index_file, index_pos, statistics
    else:
        return hidden_array_, label_, index_file, index_pos, None


def apply_kmeans(k, hi_train, hi_eval, df, proj_train, proj_eval, metadata_columns,
                 outputfile=''):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(hi_train)

    kmeans_clustering_train = kmeans.predict(hi_train)
    kmeans_clustering_eval = kmeans.predict(hi_eval)

    # UMAP
    fig = plt.figure(figsize=(15, 9))
    plt.scatter(proj_train[:, 0], proj_train[:, 1], c=kmeans_clustering_train, cmap='Set2')
    plt.scatter(proj_eval[:, 0], proj_eval[:, 1], c=kmeans_clustering_eval, cmap='Set2', marker='v')
    plt.colorbar()
    plt.tight_layout()

    plt.savefig(outputfile)
    plt.close()

    # Build dataframe with cluster information and metadata info
    df.loc[:, 'cluster'] = np.concatenate([kmeans_clustering_train, kmeans_clustering_eval])

    # Build a dataframe with percentage of each cluster per mouse x experiment
    df_gp = df.groupby(metadata_columns + ['cluster', 'train_or_test'])['cluster'].agg('count').rename('count').reset_index()
    df_count_per_GT = df.groupby(metadata_columns).agg('count').rename({'cluster': 'count'}, axis=1).reset_index()

    def norm(x):
        mask = (df_count_per_GT[metadata_columns[0]] == x[metadata_columns[0]])
        for c in metadata_columns[1:]:
            mask = mask & (df_count_per_GT[c] == x[c])
        return x['count'] / df_count_per_GT.loc[mask, 'count'].values[0]

    df_gp.loc[:, 'percent'] = df_gp.apply(norm, axis=1)

    return df, df_gp, kmeans.cluster_centers_



def plot_umaps(df, all_columns, n_components, method_type, outputdir, dataset_name, suffix):

    for j in range(0, n_components, 2):

        for label_name in all_columns:
            logging.info(f'drawing {method_type} with colors = {label_name}')

            if 'cluster' in label_name:
                fig = px.scatter(df, x=f'{method_type}_{j}', y=f'{method_type}_{j+1}', color=label_name,
                                 hover_data=all_columns + ['train_or_test'],
                                 color_continuous_scale=sns.color_palette("hls", 10).as_hex())
            else:
                fig = px.scatter(df, x=f'{method_type}_{j}', y=f'{method_type}_{j+1}', color=label_name,
                                 hover_data=all_columns + ['train_or_test'])
            fig.write_html(os.path.join(outputdir, f'{dataset_name}_normed_{method_type}_colors-{label_name}_latent-{j}{suffix}.html'))

        logging.info(f'drawing {method_type} overview')
        ncols = int(np.sqrt(len(all_columns)))
        nrows = int(np.ceil(len(all_columns) / ncols))
        fig, ax = plt.subplots(nrows, ncols, figsize=(20, 15), sharex='all', sharey='all')
        ax = ax.flatten()
        for i, label_name in enumerate(all_columns):
            if df[label_name].dtype in [float, np.float32, np.float64, int, np.int32, np.int64]:
                ax[i].set_title(
                    f'{label_name}: min {np.min(df[label_name]):.2f} - max {np.max(df[label_name]):.2f}')
                ax[i].scatter(df[f'{method_type}_{j}'], df[f'{method_type}_{j+1}'], c=df[label_name], s=1, cmap='jet')
                ax[i].set_title(label_name)
            else:
                ids, names = pd.factorize(df[label_name])
                ax[i].scatter(df[f'{method_type}_{j}'], df[f'{method_type}_{j+1}'], c=ids, s=1)
                if len(names) < 5:
                    ax[i].set_title(f'{label_name}: {" ".join([str(n) for n in names])}')
                else:
                    ax[i].set_title(f'{label_name}')

        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, f'{dataset_name}_normed_{method_type}_overview_latent-{j}{suffix}.png'))


def get_cmap(matrix, cmap_str='jet'):
    num_ = np.max(matrix)
    unique_ids = np.unique(matrix)
    unique_ids = unique_ids[unique_ids > 0]
    cmap_internal = plt.get_cmap(cmap_str)

    colors = cmap_internal([float(i) / len(unique_ids) for i in range(len(unique_ids))])
    background = "black"
    all_colors = [background if not j in unique_ids else colors[i] for i, j in enumerate(range(num_))]
    cmap_internal = ListedColormap(all_colors)
    return cmap_internal, all_colors

def plot_sequence(coordinates, skeleton_graph, keypoints, nplots, save_path, size=40, azim=60, elev=15):
    """
    Plot sequence as 3D poses, using skeleton information
    """

    min_ = np.nanmin(coordinates, axis=(0, 1))
    max_ = np.nanmax(coordinates, axis=(0, 1))
    n_dim = len(min_)

    plt.ioff()
    fig = plt.figure(figsize=(size, size), facecolor=(1, 1, 1))

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    from matplotlib import cm
    cmap = cm.get_cmap('viridis', nplots)  # N = 101

    # try to fix the hips for mouse dataset
    index_rhip = 12 #keypoints.index('leg1_0')
    index_lhip = 16 #keypoints.index('leg1_1')

    coordinates = np.dstack([coordinates[..., -1], coordinates[..., 1], coordinates[..., 0]])
    rhip_coord = coordinates[len(coordinates) // 2, index_rhip]
    lhip_coord = coordinates[len(coordinates) // 2, index_lhip]
    norm_ref = np.linalg.norm(rhip_coord - lhip_coord)
    vect_ref = (rhip_coord - lhip_coord) / norm_ref

    for idx_time in range(nplots):
        # Update 3D poses
        matrix_gt = coordinates[int(idx_time / nplots * len(coordinates))]

        # cf https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        norm_ = np.linalg.norm(matrix_gt[index_rhip] - matrix_gt[index_lhip])
        matrix_gt = (matrix_gt - matrix_gt[index_lhip])
        old_vect = matrix_gt[index_rhip] / norm_
        v = np.cross(old_vect, vect_ref)
        vx = np.array([[0, - v[2], v[1]],
                       [v[2], 0, - v[0]],
                       [- v[1], v[0], 0]])
        c = np.dot(old_vect, vect_ref)
        s = np.linalg.norm(v)
        R = np.identity(3) + vx + vx**2 * (1 - c) / s ** 2
        from scipy.spatial.transform import Rotation
        matrix_gt = Rotation.from_matrix(R).apply(matrix_gt)

        if n_dim > 2:
            ax.view_init(elev=elev, azim=azim)

        try:
            ax.set_aspect('equal')
        except NotImplementedError:
            ax.set_aspect('auto')

        for nl in skeleton_graph.neighbor_link:
            if n_dim > 2:
                ax.plot([matrix_gt[nl[0], 0], matrix_gt[nl[1], 0]],
                        [matrix_gt[nl[0], 1], matrix_gt[nl[1], 1]],
                        [matrix_gt[nl[0], 2], matrix_gt[nl[1], 2]],
                        'o-', color=cmap(idx_time))
            else:
                ax.plot([matrix_gt[nl[0], 0], matrix_gt[nl[1], 0]],
                        [matrix_gt[nl[0], 1], matrix_gt[nl[1], 1]],
                        'o-', color=cmap(idx_time))

        # ax.set_axis_off()
        plt.subplots_adjust(wspace=0, hspace=0, left=0, top=1, bottom=0, right=1)

    plt.savefig(save_path + '.svg')
    plt.savefig(save_path + '.png')

    plt.close()


def plot_sequential(coordinates, skeleton_graph, keypoints, nplots, save_path, size=40,
                  normalized_coordinates=False, azim=60):
    """
    Plot sequence as 3D poses, using skeleton information
    """

    min_ = np.nanmin(coordinates, axis=(0, 1))
    max_ = np.nanmax(coordinates, axis=(0, 1))
    n_dim = len(min_)

    plt.ioff()
    gs = gridspec.GridSpec(1, nplots, width_ratios=[1] * nplots,
                           wspace=0.0, hspace=0.0, top=1, bottom=0, left=0, right=1)
    plt.figure(figsize=(nplots * size, size), facecolor=(1, 1, 1))

    if n_dim > 2:
        index_rhip = 12 #keypoints.index('leg1_0')
        index_lhip = 16 #keypoints.index('leg1_1')
        coordinates = np.dstack([coordinates[..., -1], coordinates[..., 1], coordinates[..., 0]])
        rhip_coord = coordinates[len(coordinates) // 2, index_rhip]
        lhip_coord = coordinates[len(coordinates) // 2, index_lhip]
        norm_ref = np.linalg.norm(rhip_coord - lhip_coord)
        vect_ref = (rhip_coord - lhip_coord) / norm_ref

    for idx_time in range(nplots):
        matrix_gt = coordinates[int(idx_time / nplots * len(coordinates))]
        if n_dim > 2:
            # Update 3D poses
            norm_ = np.linalg.norm(matrix_gt[index_rhip] - matrix_gt[index_lhip])
            matrix_gt = (matrix_gt - matrix_gt[index_lhip])
            old_vect = matrix_gt[index_rhip] / norm_
            v = np.cross(old_vect, vect_ref)
            vx = np.array([[0, - v[2], v[1]],
                           [v[2], 0, - v[0]],
                           [- v[1], v[0], 0]])
            c = np.dot(old_vect, vect_ref)
            s = np.linalg.norm(v)
            R = np.identity(3) + vx + vx**2 * (1 - c) / s ** 2
            matrix_gt = Rotation.from_matrix(R).apply(matrix_gt)
            ax = plt.subplot(gs[idx_time], projection='3d')
            ax.view_init(elev=15., azim=azim)
        else:
            matrix_gt = coordinates[int(idx_time / nplots * len(coordinates))]
            ax = plt.subplot(gs[idx_time])

        try:
            ax.set_aspect('equal')
        except NotImplementedError:
            ax.set_aspect('auto')
        # if normalized_coordinates:

        if n_dim > 2:
            ax.set_xlim3d([min_[0], max_[0]])
            ax.set_ylim3d([min_[1], max_[1]])
            ax.set_zlim3d([min_[2], max_[2]])
        else:
            ax.set_xlim([min_[0], max_[0]])
            ax.set_ylim([min_[1], max_[1]])

        for nl, nlcolor in zip(skeleton_graph.neighbor_link, skeleton_graph.neighbor_link_color):
            if n_dim > 2:
                ax.plot([matrix_gt[nl[0], 0], matrix_gt[nl[1], 0]],
                        [matrix_gt[nl[0], 1], matrix_gt[nl[1], 1]],
                        [matrix_gt[nl[0], 2], matrix_gt[nl[1], 2]],
                        'o-', color=nlcolor, lw=4)
            else:
                ax.plot([matrix_gt[nl[0], 0], matrix_gt[nl[1], 0]],
                        [matrix_gt[nl[0], 1], matrix_gt[nl[1], 1]],
                        'o-', color=nlcolor, lw=4)


        ax.set_axis_off()
        plt.subplots_adjust(wspace=0, hspace=0, left=0, top=1, bottom=0, right=1)

    plt.savefig(save_path + '.svg')
    # plt.savefig(save_path + '.png')

    plt.close()



if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--checkpoint_folder", type=str, required=True)
    p.add_argument("--stride", type=float, required=True, default='in seconds')
    p.add_argument("--suffix", type=str, default='', help='string suffix added to the save files')
    p.add_argument("--k", type=int, default=10,
                   help="k for kmeans, number of wanted clusters")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format=f'[%(levelname)s][%(asctime)s] %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        handlers=[
                            logging.FileHandler(os.path.join(args.checkpoint_folder, "test_DISK_swapped_keypoints.log")),
                            logging.StreamHandler()
                        ]
                        )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numpy").setLevel(logging.WARNING)
    logging.info('Arguments:' + str(p))

    basedir = '/projects/ag-bozek/france/results_behavior'
    if not os.path.exists(basedir):
        basedir = '/home/france/Mounted_dir/results_behavior'

    config_file = os.path.join(args.checkpoint_folder, '.hydra', 'config.yaml')
    model_cfg = OmegaConf.load(config_file)
    model_path = glob(os.path.join(args.checkpoint_folder, 'model_epoch*'))[0]  # model_epoch to not take the model from the lastepoch

    dataset_constants = read_constant_file(os.path.join(basedir, 'datasets', model_cfg.dataset.name, 'constants.py'))

    if model_cfg.dataset.skeleton_file is not None:
        skeleton_file_path = os.path.join(basedir, 'datasets', model_cfg.dataset.skeleton_file)
        if not os.path.exists(skeleton_file_path):
            raise ValueError(f'no skeleton file found in', skeleton_file_path)
        skeleton_graph = Graph(file=skeleton_file_path)
    else:
        skeleton_graph = None
        skeleton_file_path = None

    """ DATA """
    model_cfg.feed_data.transforms.swap = False
    transforms, _ = init_transforms(model_cfg, dataset_constants.KEYPOINTS, dataset_constants.DIVIDER,
                                 dataset_constants.SEQ_LENGTH, basedir, args.checkpoint_folder)

    logging.info('Loading datasets...')
    train_dataset, val_dataset, test_dataset = load_datasets(dataset_name=model_cfg.dataset.name,
                                                             dataset_constants=dataset_constants,
                                                             transform=transforms,
                                                             dataset_type='full_length',
                                                             stride=args.stride,
                                                             suffix='_w-0-nans',
                                                             root_path=basedir,
                                                             length_sample=dataset_constants.SEQ_LENGTH,
                                                             freq=dataset_constants.FREQ,
                                                             outputdir=args.checkpoint_folder,
                                                             skeleton_file=None,
                                                             label_type='all',  # don't care, not using
                                                             verbose=model_cfg.feed_data.verbose)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Device: {}".format(device))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    logging.info('Loading transformer model...')
    # load model
    model = construct_NN_model(model_cfg, dataset_constants, skeleton_file_path, device)
    logging.info(f'Network constructed')

    load_checkpoint(model, None, model_path, device)

    logging.info('Extract hidden representation...')
    ### DIRECT KNN ON SEQ2SEQ LATENT SPACE
    hi_train, label_train, index_file_train, index_pos_train, statistics_train = extract_hidden(model, train_loader,
                                                                                                dataset_constants,
                                                                                                model_cfg, device,
                                                                                                compute_statistics=True)
    logging.info('Done with train hidden representation...')

    hi_eval, label_eval, index_file_eval, index_pos_eval, statistics_eval = extract_hidden(model, val_loader,
                                                                                           dataset_constants,
                                                                                           model_cfg, device,
                                                                                           compute_statistics=True)
    logging.info('Done with val hidden representation...')


    logging.info(f'hidden vectors eval {hi_eval.shape}')
    logging.info(f'hidden train eval {hi_train.shape}')

    if len(label_train.shape) > 0 and label_train.shape[1] == 4:
        label_order = label_order4
    else:
        if 'MAB' in model_cfg.dataset.name:
            label_order = ['action']
        elif 'Mocap' in model_cfg.dataset.name:
            label_order = ['name_seq', 'action']
        else:
            label_order = label_order

    ##############################################################################################
    ### Plot umap with different coloring
    #############################################################################################
    if 'FL2' in model_cfg.dataset.name:
        metadata_columns = ['experiment_str', 'experiment_type_str', 'treatment_str', 'mouse_id_str']
    else:
        try:
            metadata_columns = dataset_constants.METADATA
        except AttributeError:
            metadata_columns = []
    scalar_columns = list(statistics_train.keys()) if statistics_train is not None else []

    # Create dataframe with metdata
    df = pd.DataFrame()
    df.loc[:, 'train_or_test'] = np.concatenate([['train'] * len(label_train), ['eval'] * len(label_eval)])
    df.loc[df['train_or_test'] == 'train', 'index_file'] = index_file_train
    df.loc[df['train_or_test'] == 'eval', 'index_file'] = index_file_eval
    df.loc[df['train_or_test'] == 'train', 'index_pos'] = index_pos_train
    df.loc[df['train_or_test'] == 'eval', 'index_pos'] = index_pos_eval
    for imc, mc in enumerate(metadata_columns):
        df.loc[df['train_or_test'] == 'train', mc] = label_train[:, imc]
        df.loc[df['train_or_test'] == 'eval', mc] = label_eval[:, imc]

    print(label_order, label_train.shape)
    df.loc[:, label_order] = np.vstack([label_train, label_eval])
    # df.loc[:, 'time'] = np.concatenate([time_train, time_eval])
    if 'treatment_detail' in df.columns:
        df.loc[:, 'treatment_str'] = df['treatment_detail'].apply(lambda x: reverse_dict_treatment_detail[x])
    if 'mouse_id' in df.columns:
        df.loc[:, 'mouse_id_str'] = df['mouse_id'].astype('str')
    if 'experiment' in df.columns:
        df.loc[:, 'experiment_str'] = df['experiment'].apply(lambda x: reverse_dict_experiment[x])
    if 'experiment_type' in df.columns:
        df.loc[:, 'experiment_type_str'] = df['experiment_type'].apply(lambda x: reverse_dict_experiment_type[x])

    with_labels = False
    if 'Mocap' in model_cfg.dataset.name and 'action' in df.columns:
        reverse_dict_label = {0: 'Walk', 1: 'Wash', 2: 'Run', 3: 'Jump', 4: 'Animal Behavior', 5: 'Dance',
                                         6: 'Step', 7: 'Climb', 8: 'unknown'}
        df.loc[:, 'action_str'] = df['action'].apply(lambda x: reverse_dict_label[int(x)])
        metadata_columns += ['action_str']
        with_labels = True
    if 'MAB' in model_cfg.dataset.name and 'action' in df.columns:
        reverse_dict_label = {0: 'attack', 1: 'investigation', 2: 'mount', 3: 'other'}
        df.loc[:, 'action_str'] = df['action'].apply(lambda x: reverse_dict_label[int(x)])
        metadata_columns += ['action_str']
        with_labels = True
    all_columns = metadata_columns + scalar_columns
    logging.info(f'columns: {all_columns}')

    if statistics_train is not None:
        for key in statistics_train.keys():
            try:
                df.loc[:, key] = statistics_train[key] + statistics_eval[key]
            except ValueError as e:
                logging.warning(f'{key} NOT FOUND IN STATISICS_TRAIN.KEYS: {statistics_train.keys()}\nERROR: {e}')

    for _ in range(3):
        gc.collect()

    n_components = 4
    method = 'PCA'
    logging.info(f'Computing the {method} projection')

    myumap = PCA(n_components=n_components, svd_solver='arpack')
    if len(hi_train) > 20000:
        vect2project = hi_train[np.random.choice(np.arange(hi_train.shape[0], dtype=int), 20000, replace=False)]
    else:
        vect2project = np.array(hi_train)
    myumap.fit(vect2project)
    logging.info('Finished projecting')

    for _ in range(3):
        gc.collect()

    proj_train = myumap.transform(hi_train)
    logging.info('Finished projecting on the train')
    proj_eval = myumap.transform(hi_eval)
    logging.info('Finished projecting on the eval')
    df.loc[df['train_or_test'] == 'train', [f'{method}_{i}' for i in range(n_components)]] = proj_train
    df.loc[df['train_or_test'] == 'eval', [f'{method}_{i}' for i in range(n_components)]] = proj_eval

    logging.info(f'df columns: {df.columns}')

    plot_umaps(df, all_columns, n_components, method, args.checkpoint_folder, model_cfg.dataset.name, args.suffix)

    logging.info('Saving data in csv and npy')
    df.to_csv(os.path.join(args.checkpoint_folder, f'{model_cfg.dataset.name}{args.suffix}.csv'),
              index=False)
    columns = [c for c in df.columns if 'latent' not in c]
    df[columns].to_csv(os.path.join(args.checkpoint_folder, f'{model_cfg.dataset.name}_metadata{args.suffix}.csv'),
              index=False)
    np.save(os.path.join(args.checkpoint_folder, f'{model_cfg.dataset.name}_latent_train{args.suffix}'), hi_train)
    np.save(os.path.join(args.checkpoint_folder, f'{model_cfg.dataset.name}_latent_eval{args.suffix}'), hi_eval)
