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

    dataset_constants = read_constant_file(os.path.join('datasets', dataset_name, f'constants.py'))

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
    for id_sample in range(big_df.shape[0]):
        files = [find_file(input_folders[m], id_sample) for m in methods]
        if files == [None] * len(methods):
            # print(f'No sample found with id {id_sample}. Stopping the iteration')
            # break
            continue
        elif None in files:
            # print(f'At least one file is missing for id {id_sample}.')
            # id_sample += 1
            continue
        print(f"-- index_sample = {id_sample}")


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

    print(f'Finished with iterating the dataset')
    total_rmse = total_rmse.reset_index().convert_dtypes()
    print(f'n lines in result df: {total_rmse.groupby("method")["index"].count()}')
    print(f"RMSE per sample averaged: \n"
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
        plt.savefig(os.path.join(folder, f'{metric}_vs_gaplength_binned-5_4methods.svg'))
        plt.savefig(os.path.join(folder, f'{metric}_vs_gaplength_binned-5_4methods.png'))
        plt.close()

        plt.figure()
        plt.suptitle(metric)
        sns.lineplot(x='length_hole', y='metric_value', hue='method',
                     data=total_rmse.loc[(total_rmse['metric_type'] == metric) * (total_rmse['keypoint'] != 'all')],
                     hue_order=methods, palette=hue)
        plt.savefig(os.path.join(folder, f'{metric}_vs_gaplength_4methods.svg'))
        plt.savefig(os.path.join(folder, f'{metric}_vs_gaplength_4methods.png'))
        plt.close()

        plt.figure()
        plt.suptitle(metric)
        sns.lineplot(x='n_missing_kp', y='metric_value', hue='method',
                     data=total_rmse.loc[(total_rmse['metric_type'] == metric) * (total_rmse['keypoint'] != 'all')],
                     hue_order=methods, palette=hue)
        plt.savefig(os.path.join(folder, f'{metric}_vs_n_missing_4methods.svg'))
        plt.savefig(os.path.join(folder, f'{metric}_vs_n_missing_4methods.png'))
        plt.close()

    methods = ['DISK', 'MBI', 'kpmoseq']
    hue = ['orangered', 'gold', 'purple']

    for metric in total_rmse['metric_type'].unique():
        plt.figure()
        plt.suptitle(metric)
        sns.lineplot(x='n_missing_kp', y='metric_value', hue='method',
                     data=total_rmse.loc[(total_rmse['metric_type'] == metric) * (total_rmse['keypoint'] != 'all')],
                     hue_order=methods, palette=hue)
        plt.savefig(os.path.join(folder, f'{metric}_vs_n_missing_3methods.svg'))
        plt.savefig(os.path.join(folder, f'{metric}_vs_n_missing_3methods.png'))
        plt.close()

    methods = ['DISK', 'MBI']
    hue = ['orangered', 'gold']

    for metric in total_rmse['metric_type'].unique():
        plt.figure()
        plt.suptitle(metric)
        sns.lineplot(x='length_hole_binned', y='metric_value', hue='method',
                     data=total_rmse.loc[(total_rmse['metric_type'] == metric) * (total_rmse['keypoint'] != 'all')],
                     hue_order=methods, palette=hue)
        plt.savefig(os.path.join(folder, f'{metric}_vs_gaplength_binned-5_DISK-MBI.svg'))
        plt.savefig(os.path.join(folder, f'{metric}_vs_gaplength_binned-5_DISK-MBI.png'))
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

    plt.savefig(os.path.join(folder, 'mean_barplot_4methods.svg'))
    plt.close()


if __name__ == '__main__':

    ##########################################################################################################
    ### CHOOSE DATASET BY SUPPLYING THE COMMANDLINE ARGUMENT
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset', type=str,
                        help='dataset name', choices=['FL2', 'CLB', 'DANNCE', 'Mocap', 'DF3D', 'Fish', 'MABe', 'compare'])

    args = parser.parse_args()

    ##########################################################################################################
    ## ARGUMENTS FOR EACH DATASET

    if args.dataset == 'compare':

        mean_metrics_files = {
            'FL2': 'comparison_methods_files/FL2_comparison_methods.csv',
            'CLB': 'comparison_methods_files/CLB_comparison_methods.csv',
            'DANNCE': 'comparison_methods_files/DANNCE_comparison_methods.csv',
            'Mocap': 'comparison_methods_files/Human_comparison_methods.csv',
            'MABe': 'comparison_methods_files/MABe_comparison_methods.csv',
            'DF3D': 'comparison_methods_files/DF3D_comparison_methods.csv',
            'Fish': 'comparison_methods_files/Fish_comparison_methods.csv'}

        df = []
        for dataset, path_ in mean_metrics_files.items():
            if os.path.exists(path_):
                small_df = pd.read_csv(path_)
                print(dataset, small_df.shape)
                if small_df.shape[0] == 0:
                    continue
                small_df.loc[:, 'Dataset'] = dataset
                if 'RMSE' not in small_df.columns:
                    small_df = pd.pivot(small_df, columns='metric_type', values='metric_value',
                                        index=['method', 'repeat', 'Dataset', 'dataset']).reset_index()
                df.append(small_df)
        df = pd.concat(df)

        for metric in ['RMSE', 'MPJPE', 'PCK@0.01']:
            fig, axes = plt.subplots(1, 7, figsize=(18, 6.6))
            for i_dataset, dataset in enumerate(df['Dataset'].unique()):
                sns.barplot(data=df.loc[df['Dataset'] == dataset], x='Dataset', y=metric,
                            hue='method',
                            ax=axes[i_dataset], hue_order=['DISK', 'MBI', 'kpmoseq', 'optipose'],
                            palette=['orangered', 'gold', 'purple', 'limegreen'])

            f = plt.gcf()
            f.set_figwidth(18)
            f.set_figheight(6.6)
            plt.savefig(
                f'comparison_methods_files/barplot_comparison_{metric}.svg')
    else:
        ## FL2
        if args.dataset == 'FL2':
            ## FL2
            input_folders = {'original': 'comparison_methods_files/FL2/original',
                     'DISK': 'comparison_methods_files/FL2/DISK',
                     'MBI': 'comparison_methods_files/FL2/MBI',
                     'optipose': 'comparison_methods_files/FL2/optipose',
                     'kpmoseq': 'comparison_methods_files/FL2/kpmoseq',
            }
            output_folder = os.path.join('comparison_methods_files/FL2/comparison')
            dataset_name = 'INH_FL2_keypoints_1_60_wresiduals_w1nan_stride0.5'
            pck = 0.5684496550222218 # @0.01

        elif args.dataset == 'CLB':
            ## CLB
            input_folders = {'original': 'comparison_methods_files/CLB/original',
                     'DISK': 'comparison_methods_files/CLB/DISK',
                     'MBI': 'comparison_methods_files/CLB/MBI',
                     'optipose': 'comparison_methods_files/CLB/optipose',
                     'kpmoseq': 'comparison_methods_files/CLB/kpmoseq',
            }
            output_folder = 'comparison_methods_files/CLB/comparison'
            dataset_name = 'INH_CLB_keypoints_1_60_stride0.5'
            pck = 0.5684496550222218 # @0.01

        elif args.dataset == 'DANNCE':
            ## DANNCE
            input_folders = {'original': 'comparison_methods_files/DANNCE/original',
                     'DISK': 'comparison_methods_files/DANNCE/DISK',
                     'MBI': 'comparison_methods_files/DANNCE/MBI',
                     'optipose': 'comparison_methods_files/DANNCE/optipose',
                     'kpmoseq': 'comparison_methods_files/DANNCE/kpmoseq',
            }
            output_folder = 'comparison_methods_files/DANNCE/comparison'
            dataset_name = 'DANNCE_seq_keypoints_60_stride30_fill10'
            pck = 2.8703325891261375 # @0.01

        elif args.dataset == 'Mocap':
            ## Mocap
            input_folders = {'original': 'comparison_methods_files/Human/original',
                     'DISK': 'comparison_methods_files/Human/DISK',
                     'MBI': 'comparison_methods_files/Human/MBI',
                     'optipose': 'comparison_methods_files/Human/optipose',
                     'kpmoseq': 'comparison_methods_files/Human/kpmoseq',
            }
            output_folder = 'comparison_methods_files/Human/comparison'
            dataset_name = 'Mocap_keypoints_60_stride30'
            pck = 0.3907520187466515 # @0.01 in meters


        elif args.dataset == 'MABe':
            ## Mocap
            input_folders = {'original': 'comparison_methods_files/MABe/original',
                     'DISK': 'comparison_methods_files/MABe/DISK',
                     'MBI': 'comparison_methods_files/MABe/MBI',
                     'kpmoseq': 'comparison_methods_files/MABe/kpmoseq',
            }
            output_folder = 'comparison_methods_files/MABe/comparison'
            dataset_name = 'MABE_task1_60stride60'
            pck = 9.99527056927927 # @0.01

        elif args.dataset == 'DF3D':
            ## DF3D
            input_folders = {'original': 'comparison_methods_files/DF3D/original',
                     'DISK': 'comparison_methods_files/DF3D/DISK',
                     'optipose': 'comparison_methods_files/DF3D/optipose',
                     'MBI': 'comparison_methods_files/DF3D/MBI',
                     'kpmoseq': 'comparison_methods_files/DF3D/kpmoseq',
            }
            output_folder = 'comparison_methods_files/DF3D/comparison'
            dataset_name = 'DF3D_keypoints_60stride5'
            pck = 0.171646054776486 # @0.01

        elif args.dataset == 'Fish':
            ## Fish
            input_folders = {'original': 'comparison_methods_files/Fish/original',
                     'DISK': 'comparison_methods_files/Fish/DISK',
                     'optipose': 'comparison_methods_files/Fish/optipose',
                     'MBI': 'comparison_methods_files/Fish/MBI',
                     'kpmoseq': 'comparison_methods_files/Fish/kpmoseq',
            }
            output_folder = 'comparison_methods_files/Fish/comparison'
            dataset_name = 'Fish_v3_60stride120'
            pck = 0.171646054776486 # @0.01

        evaluate_and_plots(dataset_name, output_folder, input_folders, pck)
        plot_average(output_folder)
        plot_against_time(output_folder)
