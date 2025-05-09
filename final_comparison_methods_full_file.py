import numpy as np
import pandas as pd
import os

import matplotlib
if os.uname().nodename == 'france-XPS':
    matplotlib.use('TkAgg')
    basedir = '/home/france/Mounted_dir'
else:
    matplotlib.use('Agg')
    basedir = '/projects/ag-bozek/france'
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    imputed = np.load(
       os.path.join(basedir, 'results_behavior/datasets/INH_FL2_keypoints_1_60_wresiduals_w1nan_stride0.5_new/test_fulllength_dataset_imputed.npz'))
    orig_npz = np.load(
       os.path.join(basedir, 'results_behavior/datasets/INH_FL2_keypoints_1_60_wresiduals_w1nan_stride0.5_new/test_fulllength_dataset_w-all-nans.npz'))
    i_file = 7

    optipose = pd.read_csv(
        os.path.join(basedir, f'results_behavior/outputs/25-09-24_FL2_new_for_comparison/DISK_test/test_for_optipose_repeat_0/test_w-all-nans_file{i_file}_model_10_5_1.csv'))
    kpmoseq = pd.read_csv(
        os.path.join(basedir, f'results_behavior/outputs/25-09-24_FL2_new_for_comparison/DISK_test/test_for_optipose_repeat_0/kpmoseq/test_repeat-0_file{i_file}_kpmoseq.csv'))
    mbi = pd.read_csv(
        os.path.join(basedir, f'results_behavior/MarkerBasedImputation_FL2/model_ensemble/test_w-all-nans_file{i_file}_merged/test_w-all-nans_file{i_file}_sample0_MBI.csv'))
    n_methods = 4
    orig = pd.read_csv(
        os.path.join(basedir, f'results_behavior/outputs/25-09-24_FL2_new_for_comparison/DISK_test/test_for_optipose_repeat_0/test_w-all-nans_file{i_file}.csv'))
    output_folder = '/home/france/Dropbox/Dropbox/2021_Koeln/bogna/fig_comparison_other_methods_202502/'

    t = np.arange(3600) / 60
    x = orig_npz['X'][i_file]
    disk = imputed['X'][i_file]
    i_kp = 3  # 'left_knee'
    # KEYPOINTS = ['left_back', 'left_coord', 'left_hip', 'left_knee', 'right_back', 'right_coord', 'right_hip', 'right_knee']

    fig, axes = plt.subplots(n_methods, 3, figsize=(18, 4), sharex='all', sharey='none')
    for j in range(3):
        for i in range(n_methods):
            axes[i, j].plot(t, x[:, i_kp * 3 + j], 'o-', ms=1)

        begins = np.where(np.diff(np.isnan(x[:, i_kp * 3 + j]).astype(int)) > 0)[0]
        ends = np.where(np.diff(np.isnan(x[:, i_kp * 3 + j]).astype(int)) < 0)[0]
        if np.isnan(x[0, i_kp * 3 + j]):
            begins = np.insert(begins, 0, 0)
        for b, e in zip(begins, ends):
            for i in range(n_methods):
                axes[i, j].axvspan(xmin=t[b], xmax=t[e + 1], alpha=0.2)

            axes[0, j].plot(t[b: e + 2], disk[b:e + 2, i_kp * 3 + j], '+--', c='red')
            axes[1, j].plot(t[b: e + 2], optipose.loc[b:e + 1, f'{i_kp}_{j + 1}'].values, '+--', c='red')
            axes[2, j].plot(t[b: e + 2], kpmoseq.loc[b:e + 1, f'{i_kp}_{j + 1}'].values, '+--', c='red')
            axes[3, j].plot(t[b: e + 2], mbi.loc[b:e + 1, f'{i_kp}_{j + 1}'].values, '+--', c='red')
        begins, ends = ends, begins
        begins = np.insert(begins, 0, 0)
        if ends[0] == 0:
            ends = ends[1:]
        ends = np.insert(ends, len(ends) - 1, len(x) - 1)
        for b, e in zip(begins, ends):
            axes[0, j].plot(t[b: e + 1], disk[b:e + 1, i_kp * 3 + j], '+', alpha=0.08, c='orange')
            axes[1, j].plot(t[b: e + 1], optipose.loc[b:e, f'{i_kp}_{j + 1}'].values, '+', alpha=0.08, c='orange')
            axes[2, j].plot(t[b: e + 1], kpmoseq.loc[b:e, f'{i_kp}_{j + 1}'].values, '+', alpha=0.08, c='orange')
            axes[3, j].plot(t[b: e + 1], mbi.loc[b:e, f'{i_kp}_{j + 1}'].values, '+', alpha=0.08, c='orange')
    axes[0, 0].set_title('X')
    axes[0, 1].set_title('Y')
    axes[0, 2].set_title('Z')
    axes[2, 1].set_xlabel('time (sec)')
    axes[0, 0].set_ylabel('DISK')
    axes[1, 0].set_ylabel('Optipose')
    axes[2, 0].set_ylabel('kp-moseq')
    axes[3, 0].set_ylabel('MBI')
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_folder, f'test_w-all-nans_file{i_file}_DISK_OPTIPOSE_KPMOSEQ_MBI_kp{i_kp}_holes_shaded.png'))

    for t0, t1 in [[int(49.5 * 60), int(52.5 * 60)], [int(33.5 * 60), int(36.5 * 60)]]:
        fig, axes = plt.subplots(n_methods, 3, figsize=(18, 4), sharex='all', sharey='none')
        for j in range(3):
            for i in range(n_methods):
                axes[i, j].plot(t[t0: t1], x[t0: t1, i_kp * 3 + j], 'o-', ms=1)

            begins = np.where(np.diff(np.isnan(x[t0: t1, i_kp * 3 + j]).astype(int)) > 0)[0]
            ends = np.where(np.diff(np.isnan(x[t0: t1, i_kp * 3 + j]).astype(int)) < 0)[0]
            if np.isnan(x[t0, i_kp * 3 + j]):
                begins = np.insert(begins, 0, 0)
            for b, e in zip(begins, ends):
                for i in range(n_methods):
                    axes[i, j].axvspan(xmin=t[t0 + b], xmax=t[t0 + e + 1], alpha=0.2)

                axes[0, j].plot(t[t0 + b: t0 + e + 2], disk[t0 + b: t0 + e + 2, i_kp * 3 + j], '+--', c='red')
                axes[1, j].plot(t[t0 + b: t0 + e + 2], optipose.loc[t0 + b: t0 + e + 1, f'{i_kp}_{j + 1}'].values,
                                '+--', c='red')
                axes[2, j].plot(t[t0 + b: t0 + e + 2], kpmoseq.loc[t0 + b: t0 + e + 1, f'{i_kp}_{j + 1}'].values, '+--',
                                c='red')
                axes[3, j].plot(t[t0 + b: t0 + e + 2], mbi.loc[t0 + b: t0 + e + 1, f'{i_kp}_{j + 1}'].values, '+--',
                                c='red')
            begins, ends = ends, begins
            begins = np.insert(begins, 0, 0)
            if ends[0] == t0:
                ends = ends[1:]
            ends = np.insert(ends, len(ends) - 1, t1 - t0)
            for b, e in zip(begins, ends):
                # axes[0, j].plot(t[t0 + b: t0 + e + 1], disk[t0 + b: t0 + e + 1, i_kp * 3 + j], '+', alpha=0.4,
                #                 c='orange')
                axes[1, j].plot(t[t0 + b: t0 + e + 1], optipose.loc[t0 + b: t0 + e, f'{i_kp}_{j + 1}'].values, '+',
                                alpha=0.4, c='orange')
                axes[2, j].plot(t[t0 + b: t0 + e + 1], kpmoseq.loc[t0 + b: t0 + e, f'{i_kp}_{j + 1}'].values, '+',
                                alpha=0.4, c='orange')
                # axes[3, j].plot(t[t0 + b: t0 + e + 1], mbi.loc[t0 + b: t0 + e, f'{i_kp}_{j + 1}'].values, '+',
                #                 alpha=0.4, c='orange')
        axes[0, 0].set_title('X')
        axes[0, 1].set_title('Y')
        axes[0, 2].set_title('Z')
        axes[2, 1].set_xlabel('time (sec)')
        axes[0, 0].set_ylabel('DISK')
        axes[1, 0].set_ylabel('Optipose')
        axes[2, 0].set_ylabel('kp-moseq')
        axes[3, 0].set_ylabel('MBI')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder,
                                 f'test_w-all-nans_file{i_file}_DISK_OPTIPOSE_KPMOSEQ_MBI_kp{i_kp}_holes_shaded_{t0}.svg'))

    ######################################################################################################################################################

    imputed = np.load(
       os.path.join(basedir, 'results_behavior/datasets/INH_FL2_keypoints_1_60_wresiduals_w1nan_stride0.5_new/test_fulllength_dataset_imputed.npz'))
    orig_npz = np.load(
       os.path.join(basedir, 'results_behavior/datasets/INH_FL2_keypoints_1_60_wresiduals_w1nan_stride0.5_new/test_fulllength_dataset_w-all-nans.npz'))
    i_file = 7

    x = orig_npz['X'][i_file]
    disk = imputed['X'][i_file]
    fig, axes = plt.subplots(8, 3, figsize=(50, 8 * 3), sharex='all', sharey='none')
    for i in range(8):
        for j in range(3):
            axes[i, j].plot(x[:, i * 3 + j], 'o-', ms=1)

            begins = np.where(np.diff(np.isnan(x[:, i * 3 + j]).astype(int)) > 0)[0]
            ends = np.where(np.diff(np.isnan(x[:, i * 3 + j]).astype(int)) < 0)[0]
            if np.isnan(x[0, i * 3 + j]):
                begins = np.insert(begins, 0, 0)
            for b, e in zip(begins, ends):
                axes[i, j].axvspan(xmin=b, xmax=e + 1, alpha=0.2)
                axes[i, j].plot(np.arange(b, e + 2), disk[b:e + 2, i * 3 + j], '+--', c='red')
            # begins, ends = ends, begins
            # begins = np.insert(begins, 0, 0)
            # if ends[0] == 0:
            #     ends = ends[1:]
            # ends = np.insert(ends, len(ends) - 1, len(disk) - 1)
            # for b, e in zip(begins, ends):
            #     axes[i, j].plot(np.arange(b, e + 1), disk[b:e + 1, i * 3 + j], '+', alpha=0.1, c='orange')
    axes[0, 0].set_title('X')
    axes[0, 1].set_title('Y')
    axes[0, 2].set_title('Z')
    plt.savefig(os.path.join(output_folder, f'test_w-all-nans_file{i_file}_DISK_holes_shaded.png'))

    plt.close('all')
    optipose = pd.read_csv(
        os.path.join(basedir, f'results_behavior/outputs/25-09-24_FL2_new_for_comparison/DISK_test/test_for_optipose_repeat_0/test_w-all-nans_file{i_file}_model_10_5_1.csv'))
    orig = pd.read_csv(
        os.path.join(basedir, f'results_behavior/outputs/25-09-24_FL2_new_for_comparison/DISK_test/test_for_optipose_repeat_0/test_w-all-nans_file{i_file}.csv'))
    fig, axes = plt.subplots(8, 3, figsize=(50, 8 * 3), sharex='all', sharey='none')
    for i in range(8):
        for j in range(3):
            axes[i, j].plot(orig.loc[:, f'{i}_{j + 1}'].values, 'o-', ms=1)

            begins = np.where(np.diff(np.isnan(orig.loc[:, f'{i}_{j + 1}'].values).astype(int)) > 0)[0]
            ends = np.where(np.diff(np.isnan(orig.loc[:, f'{i}_{j + 1}'].values).astype(int)) < 0)[0]
            if np.isnan(orig.loc[:, f'{i}_{j + 1}'].values[0]):
                begins = np.insert(begins, 0, 0)
            for b, e in zip(begins, ends):
                axes[i, j].axvspan(xmin=b, xmax=e + 1, alpha=0.2)
                axes[i, j].plot(np.arange(b, e + 2), optipose.loc[b:e + 1, f'{i}_{j + 1}'].values, '+--', c='red')
            # begins, ends = ends, begins
            # begins = np.insert(begins, 0, 0)
            # if ends[0] == 0:
            #     ends = ends[1:]
            # ends = np.insert(ends, len(ends) - 1, len(disk) - 1)
            # for b, e in zip(begins, ends):
            #     axes[i, j].plot(np.arange(b, e + 1), disk[b:e + 1, i * 3 + j], '+', alpha=0.1, c='orange')
    axes[0, 0].set_title('X')
    axes[0, 1].set_title('Y')
    axes[0, 2].set_title('Z')
    plt.savefig(os.path.join(output_folder, f'test_w-all-nans_file{i_file}_OPTIPOSE_model_10_5_1_holes_shaded.png'))

    plt.close('all')
    optipose = pd.read_csv(
        os.path.join(basedir, f'results_behavior/outputs/25-09-24_FL2_new_for_comparison/DISK_test/test_for_optipose_repeat_0/kpmoseq/test_repeat-0_file{i_file}_kpmoseq.csv'))
    orig = pd.read_csv(
        os.path.join(basedir, f'results_behavior/outputs/25-09-24_FL2_new_for_comparison/DISK_test/test_for_optipose_repeat_0/test_w-all-nans_file{i_file}.csv'))
    fig, axes = plt.subplots(8, 3, figsize=(50, 8 * 3), sharex='all', sharey='none')
    for i in range(8):
        for j in range(3):
            axes[i, j].plot(orig.loc[:, f'{i}_{j + 1}'].values, 'o-', ms=1)

            begins = np.where(np.diff(np.isnan(orig.loc[:, f'{i}_{j + 1}'].values).astype(int)) > 0)[0]
            ends = np.where(np.diff(np.isnan(orig.loc[:, f'{i}_{j + 1}'].values).astype(int)) < 0)[0]
            if np.isnan(orig.loc[:, f'{i}_{j + 1}'].values[0]):
                begins = np.insert(begins, 0, 0)
            for b, e in zip(begins, ends):
                axes[i, j].axvspan(xmin=b, xmax=e + 1, alpha=0.2)
                axes[i, j].plot(np.arange(b, e + 2), optipose.loc[b:e + 1, f'{i}_{j + 1}'].values, '+--', c='red')
            # begins, ends = ends, begins
            # begins = np.insert(begins, 0, 0)
            # if ends[0] == 0:
            #     ends = ends[1:]
            # ends = np.insert(ends, len(ends) - 1, len(y) - 1)
            # for b, e in zip(begins, ends):
            #     axes[i, j].plot(np.arange(b, e + 1), y[b:e + 1, i * 3 + j], '+', alpha=0.1, c='orange')
    axes[0, 0].set_title('X')
    axes[0, 1].set_title('Y')
    axes[0, 2].set_title('Z')
    plt.savefig(os.path.join(output_folder, f'test_w-all-nans_file{i_file}_KPMOSEQ_holes_shaded.png'))
    plt.close()

    mbi = pd.read_csv(
        os.path.join(basedir, f'results_behavior/MarkerBasedImputation_FL2/model_ensemble/test_w-all-nans_file{i_file}_merged/test_w-all-nans_file{i_file}_sample0_MBI.csv'))
    orig = pd.read_csv(
        os.path.join(basedir, f'results_behavior/outputs/25-09-24_FL2_new_for_comparison/DISK_test/test_for_optipose_repeat_0/test_w-all-nans_file{i_file}.csv'))
    fig, axes = plt.subplots(8, 3, figsize=(50, 8 * 3), sharex='all', sharey='none')
    for i in range(8):
        for j in range(3):
            axes[i, j].plot(orig.loc[:, f'{i}_{j + 1}'].values, 'o-', ms=1)

            begins = np.where(np.diff(np.isnan(orig.loc[:, f'{i}_{j + 1}'].values).astype(int)) > 0)[0]
            ends = np.where(np.diff(np.isnan(orig.loc[:, f'{i}_{j + 1}'].values).astype(int)) < 0)[0]
            if np.isnan(orig.loc[:, f'{i}_{j + 1}'].values[0]):
                begins = np.insert(begins, 0, 0)
            for b, e in zip(begins, ends):
                axes[i, j].axvspan(xmin=b, xmax=e + 1, alpha=0.2)
                axes[i, j].plot(np.arange(b, e + 2), mbi.loc[b:e + 1, f'{i}_{j + 1}'].values, '+--', c='red')
            # begins, ends = ends, begins
            # begins = np.insert(begins, 0, 0)
            # if ends[0] == 0:
            #     ends = ends[1:]
            # ends = np.insert(ends, len(ends) - 1, len(y) - 1)
            # for b, e in zip(begins, ends):
            #     axes[i, j].plot(np.arange(b, e + 1), y[b:e + 1, i * 3 + j], '+', alpha=0.1, c='orange')
    axes[0, 0].set_title('X')
    axes[0, 1].set_title('Y')
    axes[0, 2].set_title('Z')
    plt.savefig(os.path.join(output_folder, f'test_w-all-nans_file{i_file}_MBI_holes_shaded.png'))
