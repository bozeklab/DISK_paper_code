import os, sys
import tqdm
from glob import glob
import importlib.util
import numpy as np
import pandas as pd
from skimage.io import imread, imsave
from functools import partial
import logging
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

import matplotlib
if os.uname().nodename == 'france-XPS':
    matplotlib.use('TkAgg')
    basedir = '/home/france/Mounted_dir/results_behavior/'
else:
    matplotlib.use('Agg')
    basedir = '/projects/ag-bozek/france/results_behavior/'

from DISK.utils.utils import read_constant_file, find_holes


def find_file(input_folder, id_sample):
    try:
        return glob(os.path.join(input_folder, f'test_repeat-0_sample{id_sample}.csv'))[0]
    except IndexError:
        try:
            return glob(os.path.join(input_folder, f'test_repeat-0_sample{id_sample}_*.csv'))[0]
        except IndexError:
            print(f'Not found `test_repeat-0_sample{id_sample}` in {input_folder}', flush=True)
            return None

def evaluate_and_plots(dataset_name, output_folder, input_folders, pck_final_threshold):

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(os.path.join(output_folder, 'plots')):
        os.mkdir(os.path.join(output_folder, 'plots'))

    dataset_constants = read_constant_file(os.path.join(basedir, 'datasets', dataset_name, f'constants.py'))

    mean_RMSE = []

    pck_name = 'PCK@0.01'

    max_n_plots = 10
    n_plots = 0

    if 'optipose' not in input_folders.keys():
        methods = ['kpmoseq', 'DISK', 'MBI']
        colors = ['purple', 'orangered', 'gold']
    else:
        methods = ['optipose', 'kpmoseq', 'DISK', 'MBI']
        colors = ['limegreen', 'purple', 'orangered', 'gold']
    # for i_repeat in range(1):
    i_repeat = 0
    total_rmse = pd.DataFrame(columns=['id_sample', 'id_hole', 'keypoint', 'method',
                                       'metric_value', 'metric_type', 'length_hole'])
    big_df = pd.read_csv(
        os.path.join(input_folders['original'], 'test_repeat-0.csv'),
        sep='|')
    id_sample = 0
    while True:
        print(f"-- index_sample = {id_sample}")
        files = [find_file(input_folders[m], id_sample) for m in methods]
        if files == [None] * len(methods):
            print(f'No sample found with id {id_sample}. Stopping the iteration')
            break
        elif None in files:
            print(f'At least one file is missing for id {id_sample}.')
            id_sample += 1
            continue
        else:
            print(f'Something wrong at id {id_sample}: {files}')
            sys.exit(1)

        list_df = [pd.read_csv(f, sep=',') for f in files]

        columns = [c for c in list_df[0].columns if c != 'behaviour']
        keypoints = [c.split('_')[0] for c in columns if c[-2:] == '_1']
        num_dims = len(columns) // len(keypoints)
        if n_plots < max_n_plots:
            orig_values = np.array(eval(big_df.loc[id_sample, 'label'])).reshape(-1, len(columns))
            orig_values_with_gap = np.array(eval(big_df.loc[id_sample, 'input'])).reshape(-1, len(columns))
            std_ = np.sum(np.std(np.sqrt(np.sum(np.diff(orig_values, axis=0)**2, axis=-1))))
            sum_mask = np.sum(orig_values_with_gap == -4668) / 3
            if std_ > 1 and sum_mask > 10:
                fig, axes = plt.subplots(len(keypoints), num_dims, figsize=(12, 9), sharex='all')#, sharey='col')
                axes = axes.flatten()
                for i in range(len(columns)):
                    axes[i].plot(orig_values[:, i], 'o-')
                    mask = orig_values_with_gap[:, i] != -4668
                    for i_method in range(len(methods)):
                        values = np.array(list_df[i_method].loc[:, columns[i]].values)
                        values[mask] = np.nan
                        axes[i].plot(values, 'o-', c=colors[i_method], label=methods[i_method], ms=4)

                axes[0].legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_folder, 'plots', f'test_repeat-0_sample{id_sample}.png'))
                plt.savefig(os.path.join(output_folder, 'plots', f'test_repeat-0_sample{id_sample}.svg'))
                plt.close()

                fig, axes = plt.subplots(len(keypoints), num_dims, figsize=(12, 9), sharex='all')#, sharey='col')
                axes = axes.flatten()
                for i in range(len(columns)):
                    axes[i].plot(orig_values[:, i], 'o-')
                    for i_method in range(len(methods)):
                        values = np.array(list_df[i_method].loc[:, columns[i]].values)
                        axes[i].plot(values, 'o-', c=colors[i_method], label=methods[i_method], ms=4)

                    ymin, ymax = axes[i].get_ylim()
                    ymean = (ymin + ymax) / 2
                    y_range = np.abs(ymax - ymin)
                    if y_range < 10:
                        axes[i].set_ylim(ymean - 5, ymean + 5)
                axes[0].legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_folder, 'plots', f'test_repeat-0_sample{id_sample}_womask.png'))
                plt.savefig(os.path.join(output_folder, 'plots', f'test_repeat-0_sample{id_sample}_womask.svg'))
                plt.close()

                n_plots += 1

        data_with_holes_np = np.array(eval(big_df.loc[id_sample, 'input']))
        values = [df.loc[:, columns].values.reshape(-1, len(keypoints), dataset_constants.DIVIDER) for df in list_df]

        full_data_np = np.array(eval(big_df.loc[id_sample, 'label']))
        mask_holes_np = (data_with_holes_np == -4668).astype(int)  #np.ones(data_with_holes_np.shape, dtype=int)#np.isnan(data_with_holes_np).astype(int)  # 1 is gap, 0 is non missing
        n_missing = np.sum(mask_holes_np[..., 0])

        list_rmse = [np.nansum(((val - full_data_np) ** 2) * mask_holes_np,
                                    axis=2) for val in values] # sum on the XYZ dimension, output shape (batch, time, keypoint)
        list_mae = [np.nansum(np.abs((val - full_data_np) * mask_holes_np),
                                    axis=2)  for val in values]# sum on the XYZ dimension, output shape (batch, time, keypoint)
        list_euclidean_distance = [np.sqrt(np.nansum(((val - full_data_np) ** 2) * mask_holes_np,
                                                          axis=2)) for val in values] # sum on the XYZ dimension, output shape (batch, time, keypoint)
        list_pck = [dist_ <= pck_final_threshold for dist_ in list_euclidean_distance]

        out = find_holes(mask_holes_np, dataset_constants.KEYPOINTS, indep=False)
        for id_hole, o in enumerate(out):  # (start, length, keypoint_name)
            slice_ = tuple([slice(o[0], o[0] + o[1], 1),
                            [dataset_constants.KEYPOINTS.index(kp) for kp in o[2].split(' ')]])
            for i_method in range(len(methods)):
                mean_euclidean = np.mean(list_euclidean_distance[i_method][slice_])
                mean_rmse = np.sqrt(np.mean(list_rmse[i_method][slice_]))
                mean_mae = np.mean(list_mae[i_method][slice_])
                mean_pck = np.sum(list_pck[i_method][slice_] * mask_holes_np[..., 0][slice_]) / np.sum(mask_holes_np[..., 0][slice_])
                total_rmse.loc[total_rmse.shape[0], :] = [id_sample, id_hole, o[2],
                                                          methods[i_method],
                                                          mean_rmse, 'RMSE',
                                                          o[1]]
                total_rmse.loc[total_rmse.shape[0], :] = [id_sample, id_hole, o[2],
                                                          methods[i_method],
                                                          mean_euclidean, 'MPJPE',
                                                          o[1]]
                total_rmse.loc[total_rmse.shape[0], :] = [id_sample, id_hole, o[2],
                                                          methods[i_method],
                                                          mean_mae, 'MAE',
                                                          o[1]]
                total_rmse.loc[total_rmse.shape[0], :] = [id_sample, id_hole, o[2],
                                                          methods[i_method],
                                                          mean_pck, pck_name,
                                                          o[1]]

        for i_method in range(len(methods)):
            total_rmse.loc[total_rmse.shape[0], :] = [id_sample, -1, 'all',
                                                      methods[i_method],
                                                      np.sum(list_pck[i_method] * mask_holes_np[..., 0]) / n_missing,
                                                      pck_name,
                                                      n_missing]
            total_rmse.loc[total_rmse.shape[0], :] = [id_sample, -1, 'all',
                                                      methods[i_method],
                                                      np.sum(list_euclidean_distance[i_method]) / n_missing,
                                                      'MPJPE',
                                                      n_missing]
            total_rmse.loc[total_rmse.shape[0], :] = [id_sample, -1, 'all',
                                                      methods[i_method],
                                                      np.sqrt(np.sum(list_rmse[i_method]) / n_missing),
                                                      'RMSE',
                                                      n_missing]
            total_rmse.loc[total_rmse.shape[0], :] = [id_sample, -1, 'all',
                                                      methods[i_method],
                                                      np.sum(list_mae[i_method]) / n_missing,
                                                      'MAE',
                                                      n_missing]
        id_sample += 1

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
    total_rmse.to_csv(os.path.join(output_folder, f'total_rmse_repeat-{i_repeat}_comparison.csv'), index=False)
    mean_df = pd.concat(mean_RMSE)
    mean_df.to_csv(os.path.join(output_folder, f'mean_rmse_comparison.csv'), index=False)


def bin_length(x, max_, bin_width):
    # bins = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250])
    bins = np.arange(0, max_, bin_width)
    middle_bins = (bins[:-1] + bins[1:]) / 2
    return middle_bins[np.argmax(x <= bins[1:])]


def plot_against_time(folder):
    total_rmse = pd.read_csv(os.path.join(folder, f'total_rmse_repeat-0_comparison.csv'))
    total_rmse.loc[:, 'length_hole_binned'] = total_rmse['length_hole'].apply(partial(bin_length, max_=total_rmse['length_hole'].max(), bin_width=5))
    total_rmse.loc[:, 'n_missing_kp'] = total_rmse['keypoint'].apply(lambda x: len(x.split(' ')))

    methods = ['DISK', 'MBI', 'kpmoseq', 'optipose']
    hue = ['orangered', 'gold', 'purple', 'limegreen']

    for metric in total_rmse['metric_type'].unique():
        plt.figure()
        plt.suptitle(metric)
        sns.lineplot(x='length_hole_binned', y='metric_value', hue='method',
                     data=total_rmse.loc[(total_rmse['metric_type'] == metric) * (total_rmse['keypoint'] != 'all')],
                     hue_order=methods, palette=hue)
        plt.savefig(os.path.join(folder, f'{metric}_vs_gaplength_binned-5_4methods_202503.svg'))
        plt.savefig(os.path.join(folder, f'{metric}_vs_gaplength_binned-5_4methods_202503.png'))
        plt.close()

        plt.figure()
        plt.suptitle(metric)
        sns.lineplot(x='length_hole', y='metric_value', hue='method',
                     data=total_rmse.loc[(total_rmse['metric_type'] == metric) * (total_rmse['keypoint'] != 'all')],
                     hue_order=methods, palette=hue)
        plt.savefig(os.path.join(folder, f'{metric}_vs_gaplength_4methods_202503.svg'))
        plt.savefig(os.path.join(folder, f'{metric}_vs_gaplength_4methods_202503.png'))
        plt.close()

        plt.figure()
        plt.suptitle(metric)
        sns.lineplot(x='n_missing_kp', y='metric_value', hue='method',
                     data=total_rmse.loc[(total_rmse['metric_type'] == metric) * (total_rmse['keypoint'] != 'all')],
                     hue_order=methods, palette=hue)
        plt.savefig(os.path.join(folder, f'{metric}_vs_n_missing_4methods_202503.svg'))
        plt.savefig(os.path.join(folder, f'{metric}_vs_n_missing_4methods_202503.png'))
        plt.close()

    methods = ['DISK', 'MBI', 'kpmoseq']
    hue = ['orangered', 'gold', 'purple']

    for metric in total_rmse['metric_type'].unique():
        plt.figure()
        plt.suptitle(metric)
        sns.lineplot(x='n_missing_kp', y='metric_value', hue='method',
                     data=total_rmse.loc[(total_rmse['metric_type'] == metric) * (total_rmse['keypoint'] != 'all')],
                     hue_order=methods, palette=hue)
        plt.savefig(os.path.join(folder, f'{metric}_vs_n_missing_3methods_202503.svg'))
        plt.savefig(os.path.join(folder, f'{metric}_vs_n_missing_3methods_202503.png'))
        plt.close()

    methods = ['DISK', 'MBI']
    hue = ['orangered', 'gold']

    for metric in total_rmse['metric_type'].unique():
        plt.figure()
        plt.suptitle(metric)
        sns.lineplot(x='length_hole_binned', y='metric_value', hue='method',
                     data=total_rmse.loc[(total_rmse['metric_type'] == metric) * (total_rmse['keypoint'] != 'all')],
                     hue_order=methods, palette=hue)
        plt.savefig(os.path.join(folder, f'{metric}_vs_gaplength_binned-5_DISK-MBI_202503.svg'))
        plt.savefig(os.path.join(folder, f'{metric}_vs_gaplength_binned-5_DISK-MBI_202503.png'))
        plt.close()


def plot_average(folder):
    mean_df = pd.read_csv(os.path.join(folder, f'mean_rmse_comparison.csv'))
    methods = ['DISK', 'MBI', 'kpmoseq', 'optipose']
    hue = ['orangered', 'gold', 'purple', 'limegreen']

    fig, axes = plt.subplots(1, 4, figsize=(10, 6))
    for i_metric, metric in enumerate(mean_df['metric_type'].unique()):
        sns.barplot(data=mean_df.loc[mean_df['metric_type'] == metric], x='metric_type', y='metric_value', hue='method',
                    ax=axes[i_metric], hue_order=methods, palette=hue)
        if 'PCK' in metric:
            axes[i_metric].set_ylim(0, 1)

    plt.savefig(os.path.join(folder, 'mean_barplot_4methods_202503.svg'))
    plt.close()


if __name__ == '__main__':

    ##########################################################################################################
    ### CHOOSE DATASET BY SUPPLYING THE COMMANDLINE ARGUMENT
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset', type=str,
                        help='dataset name', choices=['FL2', 'CLB', 'DANNCE', 'Mocap', 'DF3D', 'Fish', 'MABe'])

    args = parser.parse_args()

    ##########################################################################################################
    ## ARGUMENTS FOR EACH DATASET

    ## FL2
    if args.dataset == 'FL2':
        ## FL2
        input_folders = {'original': os.path.join(basedir, 'outputs/25-09-24_FL2_new_for_comparison/DISK_test/test_for_optipose_repeat_0'),
                 'DISK': os.path.join(basedir, 'outputs/25-09-24_FL2_new_for_comparison/DISK_test/test_for_optipose_repeat_0/DISK'),
                 'MBI': os.path.join(basedir, 'MarkerBasedImputation_FL2/model_ensemble/test_repeat-0_merged/'),
                 'optipose': os.path.join(basedir, 'outputs/25-09-24_FL2_new_for_comparison/DISK_test/test_for_optipose_repeat_0/optipose'),
                 'kpmoseq': os.path.join(basedir, 'outputs/25-09-24_FL2_new_for_comparison/DISK_test/test_for_optipose_repeat_0/kpmoseq'),
        }
        output_folder = os.path.join(basedir, 'outputs/25-09-24_FL2_new_for_comparison/DISK_test/test_for_optipose_repeat_0/comparison')
        dataset_name = 'INH_FL2_keypoints_1_60_wresiduals_w1nan_stride0.5_new'
        pck = 0.5684496550222218 # @0.01

    elif args.dataset == 'CLB':
        ## CLB
        input_folders = {'original': os.path.join(basedir, 'outputs/13-02-25_CLB_for_comparison/DISK_test/test_for_optipose_repeat_0'),
                 'DISK': os.path.join(basedir, 'outputs/13-02-25_CLB_for_comparison/DISK_test/test_for_optipose_repeat_0/DISK_pred'),
                 'MBI': os.path.join(basedir, 'MarkerBasedImputation_CLB/model_ensemble/test_repeat-0_merged/'),
                 'optipose': os.path.join(basedir, 'outputs/13-02-25_CLB_for_comparison/DISK_test/test_for_optipose_repeat_0/optipose_pred'),
                 'kpmoseq': os.path.join(basedir, 'outputs/13-02-25_CLB_for_comparison/DISK_test/test_for_optipose_repeat_0/kpmoseq'),
        }
        output_folder = os.path.join(basedir, 'outputs/13-02-25_CLB_for_comparison/DISK_test/test_for_optipose_repeat_0/comparison')
        dataset_name = 'INH_CLB_keypoints_1_60_stride0.5'
        pck = 0.5684496550222218 # @0.01

    elif args.dataset == 'DANNCE':
        ## DANNCE
        input_folders = {'original': os.path.join(basedir, 'outputs/2023-12-05_DANNCE_newnewmissing/DISK_test_for_comparison/test_for_optipose_repeat_0'),
                 'DISK': os.path.join(basedir, 'outputs/2023-12-05_DANNCE_newnewmissing/DISK_test_for_comparison/test_for_optipose_repeat_0/DISK_pred'),
                 'MBI': os.path.join(basedir, 'MarkerBasedImputation_DANNCE/model_ensemble/test_repeat-0_merged/'),
                 'optipose': os.path.join(basedir, 'outputs/2023-12-05_DANNCE_newnewmissing/DISK_test_for_comparison/test_for_optipose_repeat_0/optipose_pred'),
                 'kpmoseq': os.path.join(basedir, 'outputs/2023-12-05_DANNCE_newnewmissing/DISK_test_for_comparison/test_for_optipose_repeat_0/kpmoseq'),
        }
        output_folder = os.path.join(basedir, 'outputs/2023-12-05_DANNCE_newnewmissing/DISK_test_for_comparison/test_for_optipose_repeat_0/comparison')
        dataset_name = 'DANNCE_seq_keypoints_60_stride30_fill10_new'
        pck = 2.8703325891261375 # @0.01

    elif args.dataset == 'Mocap':
        ## Mocap
        input_folders = {'original': os.path.join(basedir, 'outputs/2025-02-24_Mocap_for_comparison/DISK_test/test_for_optipose_repeat_0'),
                 'DISK': os.path.join(basedir, 'outputs/2025-02-24_Mocap_for_comparison/DISK_test/test_for_optipose_repeat_0/DISK_pred'),
                 'MBI': os.path.join(basedir, 'MarkerBasedImputation_Mocap/model_ensemble/test_repeat-0_merged/'),
                 'optipose': os.path.join(basedir, 'outputs/2025-02-24_Mocap_for_comparison/DISK_test/test_for_optipose_repeat_0/optipose_pred'),
                 'kpmoseq': os.path.join(basedir, 'outputs/2025-02-24_Mocap_for_comparison/DISK_test/test_for_optipose_repeat_0/kpmoseq'),
        }
        output_folder = os.path.join(basedir, 'outputs/2025-02-24_Mocap_for_comparison/DISK_test/test_for_optipose_repeat_0/comparison')
        dataset_name = 'Mocap_keypoints_60_stride30_new'
        pck = 0.3907520187466515 # @0.01 in meters


    elif args.dataset == 'MABe':
        ## Mocap
        input_folders = {'original': os.path.join(basedir, 'outputs/2024-02-19_MABe_task1_newnewmissing/DISK_test/test_for_optipose_repeat_0'),
                 'DISK': os.path.join(basedir, 'outputs/2024-02-19_MABe_task1_newnewmissing/DISK_test/test_for_optipose_repeat_0/DISK_pred'),
                 'MBI': os.path.join(basedir, 'MarkerBasedImputation_MABe/model_ensemble/test_repeat-0_merged/'),
                 'kpmoseq': os.path.join(basedir, 'outputs/2024-02-19_MABe_task1_newnewmissing/DISK_test/test_for_optipose_repeat_0/kpmoseq'),
        }
        output_folder = os.path.join(basedir, 'outputs/2024-02-19_MABe_task1_newnewmissing/DISK_test/test_for_optipose_repeat_0/comparison')
        dataset_name = 'MABE_task1_60stride60'
        pck = 9.99527056927927 # @0.01

    elif args.dataset == 'DF3D':
        ## DF3D
        input_folders = {'original': os.path.join(basedir, 'outputs/2025-02-13_DF3D_for_comparison/DISK_test/test_for_optipose_repeat_0'),
                 'DISK': os.path.join(basedir, 'outputs/2025-02-13_DF3D_for_comparison/DISK_test/test_for_optipose_repeat_0/DISK_pred'),
                 'optipose': os.path.join(basedir, 'outputs/2025-02-13_DF3D_for_comparison/DISK_test/test_for_optipose_repeat_0/optipose_pred'),
                 'MBI': os.path.join(basedir, 'MarkerBasedImputation_DF3D/model_ensemble/test_repeat-0_merged/'),
                 'kpmoseq': os.path.join(basedir, 'outputs/2025-02-13_DF3D_for_comparison/DISK_test/test_for_optipose_repeat_0/kpmoseq'),
        }
        output_folder = os.path.join(basedir, 'outputs/2025-02-13_DF3D_for_comparison/DISK_test/test_for_optipose_repeat_0/comparison')
        dataset_name = 'DF3D_keypoints_60stride5_new'
        pck = 0.171646054776486 # @0.01

    elif args.dataset == 'Fish':
        ## DF3D
        input_folders = {'original': os.path.join(basedir, 'outputs/2023-09-27_Fishv3_newnewmissing/DISK_test_for_comparison/test_for_optipose_repeat_0'),
                 'DISK': os.path.join(basedir, 'outputs/2023-09-27_Fishv3_newnewmissing/DISK_test_for_comparison/test_for_optipose_repeat_0/DISK_pred'),
                 'optipose': os.path.join(basedir, 'outputs/2023-09-27_Fishv3_newnewmissing/DISK_test_for_comparison/test_for_optipose_repeat_0/optipose_pred'),
                 'MBI': os.path.join(basedir, 'MarkerBasedImputation_Fish/model_ensemble/test_repeat-0_merged/'),
                 'kpmoseq': os.path.join(basedir, 'outputs/2023-09-27_Fishv3_newnewmissing/DISK_test_for_comparison/test_for_optipose_repeat_0/kpmoseq'),
        }
        output_folder = os.path.join(basedir, 'outputs/2023-09-27_Fishv3_newnewmissing/DISK_test_for_comparison/test_for_optipose_repeat_0/comparison')
        dataset_name = 'DF3D_keypoints_60stride5_new'
        pck = 0.171646054776486 # @0.01

    else:
        sys.exit(1)

    evaluate_and_plots(dataset_name, output_folder, input_folders, pck)
    plot_average(output_folder)
    plot_against_time(output_folder)
    #
    # mean_metrics_files = {'FL2': 'outputs/25-09-24_FL2_new_for_comparison/DISK_test/test_for_optipose_repeat_0/comparison/mean_rmse_comparison.csv',
    #                   'CLB': 'outputs/13-02-25_CLB_for_comparison/DISK_test/test_for_optipose_repeat_0/comparison/mean_rmse_comparison.csv',
    #                   'DANNCE': 'outputs/2023-12-05_DANNCE_newnewmissing/DISK_test_for_comparison/test_for_optipose_repeat_0/comparison/mean_rmse_comparison.csv',
    #                   'Mocap': 'outputs/2025-02-24_Mocap_for_comparison/DISK_test/test_for_optipose_repeat_0/comparison/mean_rmse_comparison.csv',
    #                   'MABe': 'outputs/2024-02-19_MABe_task1_newnewmissing/DISK_test/test_for_optipose_repeat_0/comparison/mean_rmse_comparison.csv',
    #                   'DF3D': 'outputs/2025-02-13_DF3D_for_comparison/DISK_test/test_for_optipose_repeat_0/comparison/mean_rmse_comparison.csv',
    #                   'Fish': 'outputs/2023-09-27_Fishv3_newnewmissing/DISK_test_for_comparison/test_for_optipose_repeat_0/comparison/mean_rmse_comparison.csv'}
    #
    # df = []
    # for dataset, path_ in mean_metrics_files.items():
    #     if os.path.exists(os.path.join(basedir, path_)):
    #         small_df = pd.read_csv(os.path.join(basedir, path_))
    #         print(dataset, small_df.shape)
    #         if small_df.shape[0] == 0:
    #             continue
    #         small_df.loc[:, 'Dataset'] = dataset
    #         if 'RMSE' not in small_df.columns:
    #             small_df = pd.pivot(small_df, columns='metric_type', values='metric_value',
    #                                 index=['method', 'repeat', 'Dataset', 'dataset']).reset_index()
    #         df.append(small_df)
    # df = pd.concat(df)
    #
    # for metric in ['RMSE', 'MPJPE', 'PCK@0.01']:
    #     # print("% of improvement in terms of test RMSE between DISK and linear interpolation per dataset:\n",
    #     #       df.loc[df['Model'].isin(['linear interpolation', 'DISK'])].groupby(['dataset', 'repeat']) \
    #     #       .apply(lambda x: (x.loc[x['Model'] == 'linear interpolation', metric].values[0] - \
    #     #                         x.loc[x['Model'] == 'DISK', 'RMSE'].values[0]) /
    #     #                        x.loc[x['Model'] == 'linear interpolation', metric].values[0] * 100) \
    #     #       .groupby(['dataset']).agg(['mean', 'std']))
    #     fig, axes = plt.subplots(1, 7, figsize=(18, 6.6))
    #     for i_dataset, dataset in enumerate(df['Dataset'].unique()):
    #         sns.barplot(data=df.loc[df['Dataset'] == dataset], x='Dataset', y=metric,
    #                     hue='method',
    #                     ax=axes[i_dataset], hue_order=['DISK', 'MBI', 'kpmoseq', 'optipose'],
    #                     palette=['orangered', 'gold', 'purple', 'limegreen'])
    #
    #
    #     plt.figure()
    #     sns.barplot(data=df, x='Dataset', hue='method', y=metric,
    #                 hue_order=['DISK', 'MBI', 'kpmoseq', 'optipose'],
    #                 palette=['orangered', 'gold', 'purple', 'limegreen'])
    #     f = plt.gcf()
    #     f.set_figwidth(18)
    #     f.set_figheight(6.6)
    #     plt.savefig(f'/home/france/Dropbox/Dropbox/2021_Koeln/bogna/fig_comparison_other_methods_202502/barplot_comparison_{metric}_202503.svg')
