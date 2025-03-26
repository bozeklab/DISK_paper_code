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

from ImputeSkeleton.utils.utils import read_constant_file
from utils.experiment import BognaExp

import matplotlib
if os.uname().nodename == 'france-XPS':
    matplotlib.use('TkAgg')
    basedir = '/home/france/Mounted_dir'
else:
    matplotlib.use('Agg')
    basedir = '/projects/ag-bozek/france'


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
    file = 'FL2'
    if file == 'FL2':
        i_dim = 2
    else:
        i_dim = 0
    datafolder = os.path.join(basedir, f'results_behavior/datasets/INHCP_{file}_keypoints_60stride30/')
    data_orig = np.load(os.path.join(datafolder, 'train_fulllength_dataset_w-all-nans.npz'))
    data_imputed = np.load(os.path.join(datafolder, 'train_fulllength_dataset_imputed.npz'))
    dataset_constants = read_constant_file(os.path.join(datafolder, 'constants.py'))
    X_orig = data_orig['X'].reshape(-1, 3600, len(dataset_constants.KEYPOINTS), 3)
    X_imputed = data_imputed['X'].reshape(-1, 3600, len(dataset_constants.KEYPOINTS), 3)

    featuresdir = os.path.join(datafolder, 'bogna_features/')
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
            lambda x: [k for k, v in BognaExp.dict_label2int[exp]['mouse_id'].items() if v == x][0])
        df.loc[df['experiment_str'] == exp, 'treatment_letter'] = df.loc[
            df['experiment_str'] == exp, 'treatment'].apply(
            lambda x: [k for k, v in BognaExp.dict_label2int[exp]['treatment'].items() if v == x][0])

    # for exp in ['INH1A', 'CP1B', 'CP1A', 'INH2B']:
    #     treatments = df.loc[(df['experiment_str'] == exp), 'treatment_str'].unique()
    #     fig, axes = plt.subplots(len(treatments), 2, figsize=(10, 4 * len(treatments)))
    #     plt.suptitle(f'{exp}')
    #     for itreatment, treatment in enumerate(treatments):
    #         for side in ['left', 'right']:
    #             trajs_orig = []
    #             trajs_imputed = []
    #             n_imputed_only, n_imputed, n_orig = 0, 0, 0
    #             max_len = 0
    #             for line in df.loc[(df['experiment_str'] == exp) * (df['treatment_str'] == treatment) * (df['side'] == side)].iterrows():
    #                 index_sample = line[1]['index_sample']
    #                 start = line[1]['start']
    #                 stop = line[1]['stop']
    #                 imputed = line[1]['imputed']
    #                 ankle_marker_index = dataset_constants.KEYPOINTS.index(f'{side}_ankle')
    #                 if imputed:
    #                     n_imputed += 1
    #                     starts, stops = df.loc[(df['experiment_str'] == exp) * (df['treatment_str'] == treatment) * (df['side'] == side) * (df['imputed'] == False), ['start', 'stop']].values.T
    #                     potential_id = np.unique([np.argmin(np.abs(starts - start)), np.argmin(np.abs(stops - stop))])
    #                     duplicate = False
    #                     for pid in potential_id:
    #                         overlap = (min(stops[pid], stop) - max(starts[pid], start)) / (max(stops[pid], stop) - min(starts[pid], start))
    #                         if overlap > 0.8:
    #                             duplicate = True
    #                     if duplicate:
    #                         continue
    #                     n_imputed_only += 1
    #                     traj_ankle = X_imputed[index_sample][:, ankle_marker_index]
    #                 else:
    #                     n_orig += 1
    #                     traj_ankle = X_orig[index_sample][:, ankle_marker_index]
    #                 traj = traj_ankle[start-10:stop+10, i_dim]
    #                 vel_ankle = np.sqrt(np.sum((traj_ankle[1:, :] - traj_ankle[:-1, :]) ** 2, axis=-1)) * dataset_constants.FREQ
    #                 vel = vel_ankle[start:stop]
    #                 ## smooth by averaging over 300 / 6 = 50 frames -> 10 frames at 60Hz
    #                 smooth_window_vel = 10
    #                 smoothed_vel_ankle = np.array([np.sum(vel_ankle[i: i + smooth_window_vel]) for i in
    #                                                range(len(vel_ankle) - smooth_window_vel)]) / smooth_window_vel
    #                 padded = np.zeros(60) * np.nan
    #                 padded[:len(traj)] = traj - np.mean(traj)
    #                 if imputed:
    #                     trajs_imputed.append(padded)
    #                 else:
    #                     trajs_orig.append(padded)
    #                 if len(traj) > max_len:
    #                     max_len = len(traj)
    #
    #             mean_orig = np.nanmean(np.array(trajs_orig)[:, :max_len], axis=0)
    #             std_orig = np.nanstd(np.array(trajs_orig)[:, :max_len], axis=0)
    #             mean_imputed = np.nanmean(np.array(trajs_imputed)[:, :max_len], axis=0)
    #             std_imputed = np.nanstd(np.array(trajs_imputed)[:, :max_len], axis=0)
    #             if side == 'left':
    #                 i_ax = 0
    #             else:
    #                 i_ax = 1
    #             axes[itreatment, i_ax].set_title(f'{treatment} {side}')
    #             axes[itreatment, i_ax].plot(mean_orig, '-', c='blue', label=f'{n_orig}')
    #             axes[itreatment, i_ax].fill_between(x=np.arange(max_len), y1=mean_orig - std_orig, y2=mean_orig + std_orig, alpha=0.4,
    #                              color='blue')
    #             axes[itreatment, i_ax].plot(mean_imputed, '-', c='orange', label=f'{n_imputed_only}')
    #             axes[itreatment, i_ax].fill_between(x=np.arange(max_len), y1=mean_imputed-std_imputed, y2=mean_imputed+std_imputed, alpha=0.4, color='orange')
    #             axes[itreatment, i_ax].axvline(x=10, c='r', alpha=0.5)
    #             axes[itreatment, i_ax].legend()
    # print('stop')

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

            mean_orig = np.nanmean(np.array(trajs_orig)[:, :max_len], axis=0)
            std_orig = np.nanstd(np.array(trajs_orig)[:, :max_len], axis=0)
            mean_imputed = np.nanmean(np.array(trajs_imputed)[:, :max_len], axis=0)
            std_imputed = np.nanstd(np.array(trajs_imputed)[:, :max_len], axis=0)
            if side == 'left':
                i_ax = 0
            else:
                i_ax = 1
            axes[i_ax].set_title(f'{treatment} {side}')
            # axes[i_ax].plot(np.array(trajs_orig)[:, :max_len].T, '-', c='grey', alpha=0.1)
            axes[i_ax].plot(mean_orig, '-', c='blue', label=f'{n_orig}')
            axes[i_ax].fill_between(x=np.arange(max_len), y1=mean_orig - std_orig, y2=mean_orig + std_orig, alpha=0.4,
                             color='blue')
            axes[i_ax].plot(mean_imputed, '-', c='orange', label=f'{n_imputed_only}')
            axes[i_ax].fill_between(x=np.arange(max_len), y1=mean_imputed-std_imputed, y2=mean_imputed+std_imputed, alpha=0.4, color='orange')
            axes[i_ax].axvline(x=pad, c='r', alpha=0.5)
            axes[i_ax].legend()
            plt.suptitle(exp_name)
            axes[i_ax].set_ylim(-5, 7)
        plt.savefig(os.path.join(featuresdir, f'step_shape_{file}_{exp_name}_{treatment}_LeftRight_centeredAnklePeak.svg'))
        plt.savefig(os.path.join(featuresdir, f'step_shape_{file}_{exp_name}_{treatment}_LeftRight_centeredAnklePeak.png'))
    print('stop')