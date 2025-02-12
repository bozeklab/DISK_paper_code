import os, sys
import tqdm
from glob import glob
import importlib.util
import numpy as np
import pandas as pd
from skimage.io import imread, imsave
import logging
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns

import matplotlib
if os.uname().nodename == 'france-XPS':
    matplotlib.use('TkAgg')
    basedir = '/home/france/Mounted_dir/results_behavior/'
else:
    matplotlib.use('Agg')
    basedir = '/projects/ag-bozek/france/results_behavior/'

from DISK.utils.utils import read_constant_file


def evaluate_and_plots(dataset_name, folder):
    ###################################################################################################################

    pck_final_threshold = 0.5409094601634411
    suffix_kpmoseq = 'kpmoseq'
    suffix_optipose = 'model_10_5_1'
    suffix_DISK = 'DISK'
    suffix_MBI = 'MBI'

    ###################################################################################################################
    logging.basicConfig(level=logging.INFO,
                        format=f'[%(levelname)s][%(asctime)s] %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        filename=os.path.join(folder, f'test_comparison.log'),
                        )
    logger = logging.getLogger(__name__)

    dataset_constants = read_constant_file(os.path.join(basedir, 'datasets', dataset_name, f'constants.py'))

    mean_RMSE = []

    pck_name = 'PCK@0.01'

    max_n_plots = 50
    n_plots = 0

    for i_repeat in range(1):
        total_rmse = pd.DataFrame(columns=['id_sample', 'id_hole', 'keypoint', 'method',
                                           'metric_value', 'metric_type', 'length_hole'])
        big_df = pd.read_csv(
            os.path.join(folder, 'test_repeat-0.csv'),
            sep='|')
        for id_sample in range(0, 683):
            optipose_file = os.path.join(folder, 'optipose', f'test_repeat-0_sample{id_sample}_{suffix_optipose}.csv')
            kpmoseq_file = os.path.join(folder, 'kpmoseq', f'test_repeat-0_sample{id_sample}_{suffix_kpmoseq}.csv')
            DISK_file = os.path.join(folder, 'DISK', f'test_repeat-0_sample{id_sample}_{suffix_DISK}.csv')
            mbi_file = os.path.join(basedir, 'MarkerBasedImputation_FL2/model_ensemble/test_repeat-0_merged/', f'test_repeat-0_sample{id_sample}_{suffix_MBI}.csv')
            if not os.path.exists(optipose_file):
                continue
            original_file = os.path.join(folder, f'test_repeat-0_sample{id_sample}.csv')

            df_optipose = pd.read_csv(optipose_file, sep=',')
            df_kpmoseq = pd.read_csv(kpmoseq_file, sep=',')
            df_DISK = pd.read_csv(DISK_file, sep=',')
            df_original = pd.read_csv(original_file, sep=',')
            df_mbi = pd.read_csv(mbi_file, sep=',')

            columns = [c for c in df_optipose.columns if c != 'behaviour']
            keypoints = [c.split('_')[0] for c in columns if c[-2:] == '_1']
            num_dims = len(columns) // len(keypoints)
            if n_plots < max_n_plots:
                orig_values = np.array(eval(big_df.loc[id_sample, 'label'])).reshape(-1, len(columns))
                std_ = np.sum(np.std(np.sqrt(np.sum(np.diff(orig_values, axis=0)**2, axis=-1))))
                sum_mask = np.sum(np.isnan(df_original.loc[:, columns].values)) / 3
                if std_ > 1 and sum_mask > 10:
                    fig, axes = plt.subplots(len(keypoints), num_dims, figsize=(12, 9), sharex='all', sharey='col')
                    axes = axes.flatten()
                    for i in range(len(columns)):
                        axes[i].plot(orig_values[:, i], 'o-')
                        mask = ~np.isnan(df_original.loc[:, columns[i]].values)
                        optipose_values = np.array(df_optipose.loc[:, columns[i]].values)
                        optipose_values[mask] = np.nan
                        axes[i].plot(optipose_values, 'o-', c='limegreen', label='optipose', ms=4)
                        kpmoseq_values = np.array(df_kpmoseq.loc[:, columns[i]].values)
                        kpmoseq_values[mask] = np.nan
                        axes[i].plot(kpmoseq_values, 'o-', c='purple', label='kpmoseq', ms=4)
                        mbi_values = np.array(df_mbi.loc[:, columns[i]].values)
                        mbi_values[mask] = np.nan
                        axes[i].plot(mbi_values, 'o-', c='gold', label='mbi', ms=4)
                        disk_values = np.array(df_DISK.loc[:, columns[i]].values)
                        disk_values[mask] = np.nan
                        axes[i].plot(disk_values, 'o-', c='orangered', label='DISK', ms=4)
                        ymin, ymax = axes[i].get_ylim()
                        ymean = (ymin + ymax) / 2
                        y_range = np.abs(ymax - ymin)
                        if y_range < 10:
                            axes[i].set_ylim(ymean - 5, ymean + 5)
                    axes[0].legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(folder, 'plots', f'test_repeat-0_sample{id_sample}.png'))
                    plt.savefig(os.path.join(folder, 'plots', f'test_repeat-0_sample{id_sample}.svg'))
                    plt.close()

                    fig, axes = plt.subplots(len(keypoints), num_dims, figsize=(12, 9), sharex='all')
                    axes = axes.flatten()
                    for i in range(len(columns)):
                        axes[i].plot(orig_values[:, i], 'o-')
                        optipose_values = np.array(df_optipose.loc[:, columns[i]].values)
                        axes[i].plot(optipose_values, 'o-', c='limegreen', label='optipose', ms=4)
                        kpmoseq_values = np.array(df_kpmoseq.loc[:, columns[i]].values)
                        axes[i].plot(kpmoseq_values, 'o-', c='purple', label='kpmoseq', ms=4)
                        mbi_values = np.array(df_mbi.loc[:, columns[i]].values)
                        axes[i].plot(mbi_values, 'o-', c='gold', label='mbi', ms=4)
                        disk_values = np.array(df_DISK.loc[:, columns[i]].values)
                        axes[i].plot(disk_values, 'o-', c='orangered', label='DISK', ms=4)
                        ymin, ymax = axes[i].get_ylim()
                        ymean = (ymin + ymax) / 2
                        y_range = np.abs(ymax - ymin)
                        if y_range < 10:
                            axes[i].set_ylim(ymean - 5, ymean + 5)
                    axes[0].legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(folder, 'plots', f'test_repeat-0_sample{id_sample}_womask.png'))
                    plt.savefig(os.path.join(folder, 'plots', f'test_repeat-0_sample{id_sample}_womask.svg'))
                    plt.close()

                    n_plots += 1

            data_with_holes_np = df_original.loc[:, columns].values.reshape(-1, len(keypoints), dataset_constants.DIVIDER)
            optipose_values = df_optipose.loc[:, columns].values.reshape(-1, len(keypoints), dataset_constants.DIVIDER)
            disk_values = df_DISK.loc[:, columns].values.reshape(-1, len(keypoints), dataset_constants.DIVIDER)
            kpmoseq_values = df_kpmoseq.loc[:, columns].values.reshape(-1, len(keypoints), dataset_constants.DIVIDER)
            mbi_values = df_mbi.loc[:, columns].values.reshape(-1, len(keypoints), dataset_constants.DIVIDER)

            full_data_np = np.array(eval(big_df.loc[id_sample, 'label']))
            mask_holes_np = np.isnan(data_with_holes_np).astype(int)  #np.ones(data_with_holes_np.shape, dtype=int)#np.isnan(data_with_holes_np).astype(int)  # 1 is gap, 0 is non missing
            n_missing = np.sum(mask_holes_np[..., 0])

            rmse_optipose = np.sum(((optipose_values - full_data_np) ** 2) * mask_holes_np,
                                        axis=2)  # sum on the XYZ dimension, output shape (batch, time, keypoint)
            mae_optipose = np.sum(np.abs((optipose_values - full_data_np) * mask_holes_np),
                                        axis=2)  # sum on the XYZ dimension, output shape (batch, time, keypoint)
            euclidean_distance_optipose = np.sqrt(np.sum(((optipose_values - full_data_np) ** 2) * mask_holes_np,
                                                              axis=2))  # sum on the XYZ dimension, output shape (batch, time, keypoint)
            pck_optipose = euclidean_distance_optipose <= pck_final_threshold

            rmse_kpmoseq = np.sum(((kpmoseq_values - full_data_np) ** 2) * mask_holes_np,
                                        axis=2)  # sum on the XYZ dimension, output shape (batch, time, keypoint)
            mae_kpmoseq = np.sum(np.abs((kpmoseq_values - full_data_np) * mask_holes_np),
                                        axis=2)  # sum on the XYZ dimension, output shape (batch, time, keypoint)
            euclidean_distance_kpmoseq = np.sqrt(np.sum(((kpmoseq_values - full_data_np) ** 2) * mask_holes_np,
                                                              axis=2))  # sum on the XYZ dimension, output shape (batch, time, keypoint)
            pck_kpmoseq = euclidean_distance_kpmoseq <= pck_final_threshold

            rmse_mbi = np.sum(((mbi_values - full_data_np) ** 2) * mask_holes_np,
                                        axis=2)  # sum on the XYZ dimension, output shape (batch, time, keypoint)
            mae_mbi = np.sum(np.abs((mbi_values - full_data_np) * mask_holes_np),
                                        axis=2)  # sum on the XYZ dimension, output shape (batch, time, keypoint)
            euclidean_distance_mbi = np.sqrt(np.sum(((mbi_values - full_data_np) ** 2) * mask_holes_np,
                                                              axis=2))  # sum on the XYZ dimension, output shape (batch, time, keypoint)
            pck_mbi = euclidean_distance_mbi <= pck_final_threshold

            rmse_disk = np.sum(((disk_values - full_data_np) ** 2) * mask_holes_np,
                                        axis=2)  # sum on the XYZ dimension, output shape (batch, time, keypoint)
            mae_disk = np.sum(np.abs((disk_values - full_data_np) * mask_holes_np),
                                        axis=2)  # sum on the XYZ dimension, output shape (batch, time, keypoint)
            euclidean_distance_disk = np.sqrt(np.sum(((disk_values - full_data_np) ** 2) * mask_holes_np,
                                                              axis=2))  # sum on the XYZ dimension, output shape (batch, time, keypoint)
            pck_disk = euclidean_distance_disk <= pck_final_threshold

            total_rmse.loc[total_rmse.shape[0], :] = [id_sample, -1, 'all',
                                                      'optipose',
                                                      np.sum(pck_optipose * mask_holes_np[..., 0]) / n_missing,
                                                      pck_name,
                                                      n_missing]
            total_rmse.loc[total_rmse.shape[0], :] = [id_sample, -1, 'all',
                                                      'optipose',
                                                      np.sum(euclidean_distance_optipose) / n_missing,
                                                      'MPJPE',
                                                      n_missing]
            total_rmse.loc[total_rmse.shape[0], :] = [id_sample, -1, 'all',
                                                      'optipose',
                                                      np.sqrt(np.sum(rmse_optipose) / n_missing),
                                                      'RMSE',
                                                      n_missing]
            total_rmse.loc[total_rmse.shape[0], :] = [id_sample, -1, 'all',
                                                      'optipose',
                                                      np.sum(mae_optipose) / n_missing,
                                                      'MAE',
                                                      n_missing]

            total_rmse.loc[total_rmse.shape[0], :] = [id_sample, -1, 'all',
                                                      'kpmoseq',
                                                      np.sum(pck_kpmoseq * mask_holes_np[..., 0]) / n_missing,
                                                      pck_name,
                                                      n_missing]
            total_rmse.loc[total_rmse.shape[0], :] = [id_sample, -1, 'all',
                                                      'kpmoseq',
                                                      np.sum(euclidean_distance_kpmoseq) / n_missing,
                                                      'MPJPE',
                                                      n_missing]
            total_rmse.loc[total_rmse.shape[0], :] = [id_sample, -1, 'all',
                                                      'kpmoseq',
                                                      np.sqrt(np.sum(rmse_kpmoseq) / n_missing),
                                                      'RMSE',
                                                      n_missing]
            total_rmse.loc[total_rmse.shape[0], :] = [id_sample, -1, 'all',
                                                      'kpmoseq',
                                                      np.sum(mae_kpmoseq) / n_missing,
                                                      'MAE',
                                                      n_missing]

            total_rmse.loc[total_rmse.shape[0], :] = [id_sample, -1, 'all',
                                                      'mbi',
                                                      np.sum(pck_mbi * mask_holes_np[..., 0]) / n_missing,
                                                      pck_name,
                                                      n_missing]
            total_rmse.loc[total_rmse.shape[0], :] = [id_sample, -1, 'all',
                                                      'mbi',
                                                      np.sum(euclidean_distance_mbi) / n_missing,
                                                      'MPJPE',
                                                      n_missing]
            total_rmse.loc[total_rmse.shape[0], :] = [id_sample, -1, 'all',
                                                      'mbi',
                                                      np.sqrt(np.sum(rmse_mbi) / n_missing),
                                                      'RMSE',
                                                      n_missing]
            total_rmse.loc[total_rmse.shape[0], :] = [id_sample, -1, 'all',
                                                      'mbi',
                                                      np.sum(mae_mbi) / n_missing,
                                                      'MAE',
                                                      n_missing]

            total_rmse.loc[total_rmse.shape[0], :] = [id_sample, -1, 'all',
                                                      'DISK',
                                                      np.sum(pck_disk * mask_holes_np[..., 0]) / n_missing,
                                                      pck_name,
                                                      n_missing]
            total_rmse.loc[total_rmse.shape[0], :] = [id_sample, -1, 'all',
                                                      'DISK',
                                                      np.sum(euclidean_distance_disk) / n_missing,
                                                      'MPJPE',
                                                      n_missing]
            total_rmse.loc[total_rmse.shape[0], :] = [id_sample, -1, 'all',
                                                      'DISK',
                                                      np.sqrt(np.sum(rmse_disk) / n_missing),
                                                      'RMSE',
                                                      n_missing]
            total_rmse.loc[total_rmse.shape[0], :] = [id_sample, -1, 'all',
                                                      'DISK',
                                                      np.sum(mae_disk) / n_missing,
                                                      'MAE',
                                                      n_missing]

        logging.info(f'Finished with iterating the dataset')
        total_rmse = total_rmse.reset_index().convert_dtypes()
        logging.info(f'n lines in result df: {total_rmse.shape[0]}')
        logging.info(f"RMSE per sample averaged: \n"
                     f"{total_rmse[(total_rmse['metric_type'].isin([pck_name, 'RMSE', 'MPJPE', 'MAE'])) * (total_rmse['keypoint'] == 'all')].groupby(['metric_type', 'method'])['metric_value'].agg('mean')}")
        tmp = total_rmse[
            (total_rmse['metric_type'].isin([pck_name, 'RMSE', 'MPJPE', 'MAE'])) * (total_rmse['keypoint'] == 'all')].groupby(
            ['metric_type', 'method'])['metric_value'].agg('mean').reset_index()
        tmp['repeat'] = i_repeat
        tmp['dataset'] = dataset_name
        mean_RMSE.append(tmp)
        total_rmse.to_csv(os.path.join(folder, f'total_rmse_repeat-{i_repeat}_comparison.csv'), index=False)
    mean_df = pd.concat(mean_RMSE)
    mean_df.to_csv(os.path.join(folder, f'mean_rmse_comparison.csv'), index=False)

def plot_average(folder):
    mean_df = pd.read_csv(os.path.join(folder, f'mean_rmse_comparison.csv'))
    methods = ['DISK', 'mbi', 'kpmoseq', 'optipose']
    hue = ['orangered', 'gold', 'purple', 'limegreen']
    fig, axes = plt.subplots(1, 4, figsize=(10, 6))
    sns.barplot(data=mean_df.loc[mean_df['metric_type'] == 'MAE'], x='metric_type', y='metric_value', hue='method', ax=axes[0], hue_order=methods, palette=hue)
    sns.barplot(data=mean_df.loc[mean_df['metric_type'] == 'RMSE'], x='metric_type', y='metric_value', hue='method', ax=axes[1], hue_order=methods, palette=hue)
    sns.barplot(data=mean_df.loc[mean_df['metric_type'] == 'MPJPE'], x='metric_type', y='metric_value', hue='method', ax=axes[2], hue_order=methods, palette=hue)
    sns.barplot(data=mean_df.loc[mean_df['metric_type'] == 'PCK@0.01'], x='metric_type', y='metric_value', hue='method', ax=axes[3], hue_order=methods, palette=hue)
    axes[3].set_ylim(0, 1)
    plt.savefig(os.path.join(folder, 'mean_barplot_4methods_202502.svg'))
    plt.close()


if __name__ == '__main__':
    folder = os.path.join(basedir, 'outputs/25-09-24_FL2_new_for_comparison/DISK_test/test_for_optipose_repeat_0')
    dataset_name = 'INH_FL2_keypoints_1_60_wresiduals_w1nan_stride0.5_new'

    evaluate_and_plots(dataset_name, folder)
    plot_average(folder)