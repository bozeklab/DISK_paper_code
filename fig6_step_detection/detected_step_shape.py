import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from glob import glob
from scipy.signal import find_peaks, medfilt
from scipy.interpolate import Akima1DInterpolator
import seaborn as sns
from scipy.io import loadmat

from DISK.utils.utils import read_constant_file

import matplotlib
matplotlib.use('Agg')

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
                  }


def get10sec_window(x):
    freq = 60
    if x <= freq * 10:
        return 0
    elif x <= freq * 20:
        return 1
    elif x <= freq * 30:
        return 2
    elif x <= freq * 40:
        return 3
    elif x <= freq * 50:
        return 4
    else:
        return 5


if __name__ == '__main__':
    i_dim = 2

    datafolder = '../datasets/INHCP_FL2_keypoints_60stride30/'
    data_orig = np.load(os.path.join(datafolder, 'train_fulllength_dataset_w-all-nans.npz'))
    data_imputed = np.load(os.path.join(datafolder, 'train_fulllength_dataset_imputed.npz'))
    dataset_constants = read_constant_file(os.path.join(datafolder, 'constants.py'))
    X_orig = data_orig['X'].reshape(-1, 3600, len(dataset_constants.KEYPOINTS), 3)
    X_imputed = data_imputed['X'].reshape(-1, 3600, len(dataset_constants.KEYPOINTS), 3)

    featuresdir = '../datasets/INHCP_FL2_keypoints_60stride30/step_features'
    df_before = pd.read_csv(os.path.join(featuresdir, 'swing_properties_swing_duration_train_fulllength.csv'))
    df_after = pd.read_csv(os.path.join(featuresdir, 'swing_properties_swing_duration_train_fulllength_imputed.csv'))

    df_after.loc[:, 'imputed'] = True
    df_before.loc[:, 'imputed'] = False
    df = pd.concat([df_before, df_after])
    df.loc[:, 'side_imputed'] = df[['side', 'imputed']].apply(lambda x: f'{x[0]}_{["orig", "imputed"][x[1]]}', axis=1)
    df = df.reset_index()
    df.loc[:, '10second_window'] = df['vel_peak_index'].apply(get10sec_window)
    df.loc[:, 'window_10sec'] = df['10second_window'].astype(str)
    for exp in ['INH1A', 'CP1B', 'CP1A', 'INH2B']:
        if exp == 'CP1A':
            exp = 'CP1'
        df.loc[df['experiment_str'] == exp, 'mouse_id_str'] = df.loc[df['experiment_str'] == exp, 'mouse_id'].apply(
            lambda x: [k for k, v in dict_label2int[exp]['mouse_id'].items() if v == x][0])
        df.loc[df['experiment_str'] == exp, 'treatment_letter'] = df.loc[
            df['experiment_str'] == exp, 'treatment_detail'].apply(
            lambda x: [k for k, v in dict_label2int[exp]['treatment_detail'].items() if v == x][0])

    sns.set_style('ticks')
    pad = 5
    # for treatment in ['vehicle_INH', 'vehicle_CP']:
    treatment = 'vehicle_INH'
    for exp in [['INH1A'], ['INH2B'], ['INH1A', 'INH2B', ]]:
        exp_name = exp[0] if len(exp) == 1 else '+'.join(exp)
        fig, axes = plt.subplots(1, 2, figsize=(6, 4))
        for side in ['left', 'right']:
            trajs_orig = []
            trajs_imputed = []
            n_imputed_only, n_imputed, n_orig = 0, 0, 0
            max_len = 0
            # for line in df.loc[(df['treatment_str'] == treatment)].iterrows():
            for line in df.loc[(df['experiment_str'].isin(exp)) * (df['treatment_str'] == treatment) * (df['side'] == side)].iterrows():
                index_sample = line[1]['index_sample']
                vel_peak_index = line[1]['vel_peak_index']
                start = line[1]['start']
                stop = line[1]['stop']
                imputed = line[1]['imputed']
                ankle_marker_index = dataset_constants.KEYPOINTS.index(f'{side}_ankle')
                if imputed:
                    n_imputed += 1
                    # starts, stops = df.loc[(df['treatment_str'] == treatment) * (df['imputed'] == False), ['start', 'stop']].values.T
                    starts, stops = df.loc[(df['experiment_str'].isin(exp)) * (df['treatment_str'] == treatment) * (df['side'] == side) * (df['imputed'] == False), ['start', 'stop']].values.T
                    potential_id = np.unique([np.argmin(np.abs(starts - start)), np.argmin(np.abs(stops - stop))])
                    duplicate = False
                    for pid in potential_id:
                        overlap = (min(stops[pid], stop) - max(starts[pid], start)) / (max(stops[pid], stop) - min(starts[pid], start))
                        if overlap > 0.8:
                            duplicate = True
                    if duplicate:
                        continue
                    n_imputed_only += 1
                    traj_ankle = X_imputed[index_sample][:, ankle_marker_index]
                else:
                    n_orig += 1
                    traj_ankle = X_orig[index_sample][:, ankle_marker_index]
                peak, _ = find_peaks(traj_ankle[start:stop, i_dim])
                if len(peak) > 1:
                    p = np.argmax(traj_ankle[start + peak, i_dim])
                    peak = peak[p]
                elif len(peak) == 1:
                    peak = peak[0]
                else:
                    continue
                traj = traj_ankle[max(0, start + peak-pad): start + peak+pad, i_dim]
                vel_ankle = np.sqrt(np.sum((traj_ankle[1:, :] - traj_ankle[:-1, :]) ** 2, axis=-1)) * dataset_constants.FREQ
                vel = vel_ankle[start + peak - pad: start + peak + pad]
                ## smooth by averaging over 300 / 6 = 50 frames -> 10 frames at 60Hz
                smooth_window_vel = 10
                smoothed_vel_ankle = np.array([np.sum(vel_ankle[i: i + smooth_window_vel]) for i in
                                               range(len(vel_ankle) - smooth_window_vel)]) / smooth_window_vel
                padded = np.zeros(2 * pad) * np.nan
                mean_ = np.mean(traj)
                max_ = np.max(traj)
                min_ = np.min(traj)
                padded[:len(traj)] = traj - mean_ #(traj - min_) / (max_ - min_ + 1e-9)
                if imputed:
                    trajs_imputed.append(padded)
                else:
                    trajs_orig.append(padded)
                if len(traj) > max_len:
                    max_len = len(traj)

            print(f'{exp} - {side} - Using {np.array(trajs_orig).shape[0]} ORIG detected steps and {np.array(trajs_imputed).shape[0]} for the plot')
            mean_orig = np.nanmean(np.array(trajs_orig)[:, :max_len], axis=0)
            std_orig = np.nanstd(np.array(trajs_orig)[:, :max_len], axis=0)
            mean_imputed = np.nanmean(np.array(trajs_imputed)[:, :max_len], axis=0)
            std_imputed = np.nanstd(np.array(trajs_imputed)[:, :max_len], axis=0)
            if side == 'left':
                i_ax = 0
            else:
                i_ax = 1
            axes[i_ax].set_title(f'{treatment} {side}')
            axes[i_ax].plot(mean_orig, '-', c='blue', label=f'{n_orig}')
            axes[i_ax].fill_between(x=np.arange(max_len), y1=mean_orig - std_orig, y2=mean_orig + std_orig, alpha=0.4,
                             color='blue')
            axes[i_ax].plot(mean_imputed, '-', c='orange', label=f'{n_imputed_only}')
            axes[i_ax].fill_between(x=np.arange(max_len), y1=mean_imputed-std_imputed, y2=mean_imputed+std_imputed, alpha=0.4, color='orange')
            axes[i_ax].axvline(x=pad, c='r', alpha=0.5)
            axes[i_ax].legend()
            plt.suptitle(exp_name)
            axes[i_ax].set_ylim(-5, 7)
        plt.savefig(os.path.join(featuresdir, f'step_shape_FL2_{exp_name}_{treatment}_LeftRight_centeredAnklePeak.svg'))
        plt.savefig(os.path.join(featuresdir, f'step_shape_FL2_{exp_name}_{treatment}_LeftRight_centeredAnklePeak.png'))
    print('stop')

"""
OUTPUTS

['INH1A'] - left - Using 311 ORIG detected steps and 37 for the plot
['INH1A'] - right - Using 286 ORIG detected steps and 57 for the plot
['INH2B'] - left - Using 202 ORIG detected steps and 125 for the plot
['INH2B'] - right - Using 158 ORIG detected steps and 152 for the plot
['INH1A', 'INH2B'] - left - Using 513 ORIG detected steps and 142 for the plot
['INH1A', 'INH2B'] - right - Using 444 ORIG detected steps and 181 for the plot

"""