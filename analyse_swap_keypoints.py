import os, sys
import importlib
from functools import partial
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_average(output_dir, df, dataset_name):

    renamemethods_dict = {'linear_interp': 'linear interpolation',
                          # 'mask-False_swap-0.1_2': 'DISK Swap woMask',
                          # 'mask-False_swap-0_0': 'DISK woSwap woMask',
                          # 'mask-False_swap-0_4': '',
                          # 'mask-False_swap-0_3': '',
                          # 'mask-True_swap-0.1_0': 'DISK Swap',
                          'mask-True_swap-0.1_0': 'DISK Swap',
                          'mask-True_swap-0.1_1': 'DISK Swap',
                          'mask-True_swap-0_0': 'DISK woSwap',
                          'mask-True_swap-0_1': 'DISK woSwap'}
    methods_order = ['linear interpolation', 'DISK woSwap', 'DISK Swap']
    palette = ['gray', 'gold', 'orangered']

    df.loc[:, 'method'] = df['method_param'].apply(lambda x: renamemethods_dict[x])

    print(df.groupby(['metric_type', 'swap', 'method'])['metric_value'].mean())

    fig, axes = plt.subplots(1, 3, sharey='col', figsize=(8, 4))
    sns.barplot(data=df.loc[df['metric_type'] == 'RMSE'], x='swap', y='metric_value', hue='method',
                hue_order=methods_order,
                palette=palette, ax=axes[0])
    sns.barplot(data=df.loc[df['metric_type'] == 'PCK@0.01'], x='swap', y='metric_value',
                hue='method',
                hue_order=methods_order,
                palette=palette, ax=axes[1])
    axes[1].set_ylim(0, 1.)
    plt.legend([])
    sns.barplot(data=df.loc[df['metric_type'] == 'MPJPE'], x='swap', y='metric_value',
                hue='method',
                hue_order=methods_order,
                palette=palette, ax=axes[2])
    plt.legend([])
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f'barplot_swap_{dataset_name}_origccords.svg'))
    plt.savefig(
        os.path.join(output_dir, f'barplot_swap_{dataset_name}_origccords.png'))


def bin_length(x, max_, min_, bin_width=None, n_bins=None):
    if n_bins is None and bin_width is None:
        return ValueError(f'[bin_length function] One of the two parameters `bin_width` or `n_bins` should not be None.')
    # bins = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250])
    if bin_width is not None:
        bins = np.arange(min_, max_, bin_width)
    else:
        bins = np.arange(min_, max_, (max_ - min_) / n_bins)
    middle_bins = (bins[:-1] + bins[1:]) / 2
    return middle_bins[np.argmax(x <= bins[1:])]


if __name__ == '__main__':
    ##########################################################################################################
    ### CODE FOR FIGURES FOR SWAP KEYPOINTS
    ### unzip swap_files.zip
    ##########################################################################################################

    ##########################################################################################################
    ### CHOOSE DATASET BY SUPPLYING THE COMMANDLINE ARGUMENT
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset', type=str,
                        help='dataset name', choices=['FL2', 'CLB', 'DANNCE', 'Mocap', 'DF3D', 'Fish', 'MABe', 'compare'])

    args = parser.parse_args()

    ##########################################################################################################
    ## ARGUMENTS FOR EACH DATASET
    output_dir = '/home/france/Dropbox/Dropbox/2021_Koeln/bogna/chosen_plots_swap_202502/'

    if args.dataset == 'MABe':
        dataset_name = 'MABE_task1_60stride60'
        input_dir = 'swap_files/2025-03-20_test_compare_MABe_SWAP0.5'

    elif args.dataset == 'FL2':
        dataset_name = 'INH_FL2_keypoints_1_60_wresiduals_w1nan_stride0.5'
        input_dir = 'swap_files/2025-03-20_test_compare_FL2_SWAP0.5'

    elif args.dataset == 'CLB':
        dataset_name = 'INH_CLB_keypoints_1_60_stride0.5'
        input_dir = 'swap_files/2025-03-20_test_compare_CLB_SWAP0.5'

    elif args.dataset == 'DANNCE': ## MISSING
        dataset_name = 'DANNCE_seq_keypoints_60_stride30_fill10'
        input_dir = 'swap_files/2025-03-20_test_compare_DANNCE_SWAP0.5'

    elif args.dataset == 'Mocap':
        dataset_name = 'Mocap_keypoints_60_stride30'
        input_dir = 'swap_files/2025-03-20_test_compare_Mocap_SWAP0.5'

    elif args.dataset == 'DF3D': ## MISSING
        dataset_name = 'DF3D_keypoints_60stride5'
        input_dir = 'swap_files/2025-03-20_test_compare_DF3D_SWAP0.5'

    elif args.dataset == 'Fish':
        dataset_name = 'Fish_v3_60stride120'
        input_dir = 'swap_files/2025-03-20_test_compare_Fish_SWAP0.5'

    else:
        sys.exit(1)

    df = pd.read_csv(os.path.join(input_dir, 'total_metrics_repeat-0.csv'))
    print(df.loc[:, 'method_param'].unique())

    list_no_swap = ['(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)', '(0, 1, 2, 3, 4, 5, 6, 7)',
                    '(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)',
                    '(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37)',
                    '(0, 1, 2, 3, 4, 5)']
    df.loc[:, 'swap'] = df['swap_kp_id'].apply(lambda x: False if x in list_no_swap  else True)

    plot_average(output_dir, df, args.dataset)

    skeleton_file = f'datasets/{dataset_name}/skeleton.py'
    spec = importlib.util.spec_from_file_location("module.name", skeleton_file)
    skeleton_inputs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(skeleton_inputs)
    keypoints = skeleton_inputs.keypoints

    df_swap = df.loc[df['swap'] == True, :]

    df_swap.loc[:, 'swap_kp_id'] = df_swap['swap_kp_id'].apply(eval)
    df_swap.loc[:, 'swap_kp_id0'] = df_swap['swap_kp_id'].apply(lambda x: keypoints[x[0]])
    df_swap.loc[:, 'swap_kp_id1'] = df_swap['swap_kp_id'].apply(lambda x: keypoints[x[1]])
    df_swap.loc[:, 'length_hole_binned'] = df_swap['length_hole'].apply(partial(bin_length, max_=df_swap['length_hole'].max(), min_=0, bin_width=5))
    df_swap.loc[:, 'swap_length_binned'] = df_swap['swap_length'].apply(partial(bin_length, max_=df_swap['swap_length'].max(), min_=0, bin_width=5))
    df_swap.loc[:, 'average_dist_bw_swap_kp_binned'] = df_swap['average_dist_bw_swap_kp'].apply(partial(bin_length,
                                                                                                        max_=df_swap['average_dist_bw_swap_kp'].max(),
                                                                                                        min_=df_swap['average_dist_bw_swap_kp'].min(),
                                                                                                        n_bins=30))
    plt.close('all')

    df_swap_sym = df_swap.copy(deep=True)
    df_swap_sym.loc[:, 'swap_kp_id0'] = df_swap['swap_kp_id1'].values
    df_swap_sym.loc[:, 'swap_kp_id1'] = df_swap['swap_kp_id0'].values
    df_swap = pd.concat([df_swap, df_swap_sym])

    for metric in df_swap['metric_type'].unique():
        if 'uncertainty' in metric or 'sigma' in metric:
            continue

        mean_metric_df = df_swap.loc[df_swap['metric_type'] == metric].groupby(['swap_kp_id0', 'swap_kp_id1'])['metric_value'].mean().reset_index()
        mean_metric_df2 = df_swap.loc[df_swap['metric_type'] == metric].groupby(['swap_kp_id0', 'swap_kp_id1'])['metric_value'].std().reset_index().rename(columns={'metric_value': 'std'})


        mean_metric_df = pd.merge(mean_metric_df, mean_metric_df2, on=['swap_kp_id0', 'swap_kp_id1'], how='left')
        mean_metric_df.swap_kp_id0 = mean_metric_df.swap_kp_id0.astype("category")
        mean_metric_df.swap_kp_id0 = mean_metric_df.swap_kp_id0.cat.set_categories(keypoints)

        mean_metric_df.swap_kp_id1 = mean_metric_df.swap_kp_id1.astype("category")
        mean_metric_df.swap_kp_id1 = mean_metric_df.swap_kp_id1.cat.set_categories(keypoints)

        mat = mean_metric_df.pivot(index='swap_kp_id0', columns='swap_kp_id1', values='metric_value')
        mat = mat.reindex(keypoints)
        for k in keypoints:
            if k not in mat.columns:
                mat[k] = np.nan
        mat = mat[keypoints]

        plt.figure(figsize=(12, 12))
        # sns.barplot(df_swap.loc[(df_swap['metric_type'] == metric) * (df_swap['keypoint'] != 'all')], y='swap_kp_id0', x='metric_value', hue='swap_kp_id1', hue_order=keypoints, order=keypoints, palette='Set2')
        sns.heatmap(mat, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f", )
        plt.savefig(os.path.join(input_dir, f'heatmap_{args.dataset}_swap_kp_id_{metric}.png'))
        plt.savefig(os.path.join(input_dir, f'heatmap_{args.dataset}_swap_kp_id_{metric}.svg'))
        plt.close('all')

        plt.figure(figsize=(12, 12))
        sns.barplot(df_swap.loc[(df_swap['metric_type'] == metric) * (df_swap['keypoint'] != 'all')], y='swap_kp_id0', x='metric_value', hue='swap_kp_id1', hue_order=keypoints, order=keypoints, palette='Set2')
        # sns.heatmap(mat, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f", )
        plt.savefig(os.path.join(input_dir, f'barplot_{args.dataset}_swap_kp_id_{metric}.png'))
        plt.savefig(os.path.join(input_dir, f'barplot_{args.dataset}_swap_kp_id_{metric}.svg'))
        plt.close('all')

        mean_metric_df = df_swap.loc[(df_swap['metric_type'] == metric) * (df_swap['keypoint'] != 'all'),
        ['metric_value', 'length_hole_binned', 'swap_length_binned']].groupby(['length_hole_binned', 'swap_length_binned']).mean().reset_index()
        mat_length = mean_metric_df.pivot(index='length_hole_binned', columns='swap_length_binned', values='metric_value')
        mat_length = mat_length.reindex(list(mat_length.index)[::-1])

        plt.figure(figsize=(12, 12))
        plt.suptitle(metric)
        sns.heatmap(mat_length, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f", )
        plt.savefig(os.path.join(input_dir, f'heatmap_{args.dataset}_swap_length_{metric}.png'))
        plt.savefig(os.path.join(input_dir, f'heatmap_{args.dataset}_swap_length_{metric}.svg'))
        plt.close('all')

        sns.lineplot(data=df_swap.loc[(df_swap['metric_type'] == metric) * (df_swap['keypoint'] != 'all')],
                     x='average_dist_bw_swap_kp_binned', y='metric_value')
        plt.savefig(os.path.join(input_dir, f'lineplot_{args.dataset}_swap_dist_bw_kp_{metric}.png'))
        plt.savefig(os.path.join(input_dir, f'lineplot_{args.dataset}_swap_dist_bw_kp_{metric}.svg'))