import os, sys
import tqdm
from glob import glob

import numpy as np
import pandas as pd
from skimage.io import imread, imsave
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib
if os.uname().nodename == 'france-XPS':
    matplotlib.use('TkAgg')
    basedir = '/home/france/Mounted_dir/results_behavior'
else:
    matplotlib.use('Agg')
    basedir = '/projects/ag-bozek/france/results_behavior'


def plot_average():
    df_mean = pd.read_csv(os.path.join(basedir, 'outputs/test_compare_MABe_SWAP_origcoord/mean_metrics.csv'))

    renamemethods_dict = {'linear_interp': 'linear interpolation',
                          # 'mask-False_swap-0.1_2': 'DISK Swap woMask',
                          # 'mask-False_swap-0_0': 'DISK woSwap woMask',
                          # 'mask-False_swap-0_4': '',
                          'mask-True_swap-0.1_0': 'DISK Swap',
                          'mask-True_swap-0_1': 'DISK woSwap'}
    methods_order = ['linear_interp', 'DISK woSwap', 'DISK Swap']
    palette = ['grey', 'red', 'orangered']

    df_mean.loc[:, 'method'] = df_mean['method_param'].apply(lambda x: renamemethods_dict[x])

    fig, axes = plt.subplots(1, 3, sharey='col', figsize=(12, 6))
    sns.barplot(data=df_mean.loc[df_mean['metric_type'] == 'RMSE'], x='metric_type', y='metric_value', hue='method',
                hue_order=methods_order,
                palette=palette, ax=axes[0])
    sns.barplot(data=df_mean.loc[df_mean['metric_type'] == 'PCK@0.01'], x='metric_type', y='metric_value',
                hue='method',
                hue_order=methods_order,
                palette=palette, ax=axes[1])
    axes[1].set_ylim(0, 1.)
    plt.legend([])
    sns.barplot(data=df_mean.loc[df_mean['metric_type'] == 'MPJPE'], x='metric_type', y='metric_value',
                hue='method',
                hue_order=methods_order,
                palette=palette, ax=axes[2])
    plt.legend([])
    plt.savefig(
        '/home/france/Dropbox/Dropbox/2021_Koeln/bogna/chosen_plots_swap_202502/barplot_swap_MABe_origccords.svg')
    plt.savefig(
        '/home/france/Dropbox/Dropbox/2021_Koeln/bogna/chosen_plots_swap_202502/barplot_swap_MABe_origccords.png')

    df_mean = pd.read_csv(
        '/home/france/Mounted_dir/results_behavior/outputs/test_compare_MABe_SWAP/mean_metrics.csv')
    df_mean.loc[:, 'method'] = df_mean['method_param'].apply(lambda x: renamemethods_dict[x])

    fig, axes = plt.subplots(1, 3, sharey='col', figsize=(12, 6))
    sns.barplot(data=df_mean.loc[df_mean['metric_type'] == 'RMSE'], x='metric_type', y='metric_value', hue='method',
                hue_order=methods_order,
                palette=palette, ax=axes[0])
    sns.barplot(data=df_mean.loc[df_mean['metric_type'] == 'PCK@0.01'], x='metric_type', y='metric_value',
                hue='method',
                hue_order=methods_order,
                palette=palette, ax=axes[1])
    axes[1].set_ylim(0, 1.)
    plt.legend([])
    sns.barplot(data=df_mean.loc[df_mean['metric_type'] == 'MPJPE'], x='metric_type', y='metric_value',
                hue='method',
                hue_order=methods_order,
                palette=palette, ax=axes[2])
    plt.legend([])
    plt.savefig('/home/france/Dropbox/Dropbox/2021_Koeln/bogna/chosen_plots_swap_202502/barplot_swap_MABe.svg')
    plt.savefig('/home/france/Dropbox/Dropbox/2021_Koeln/bogna/chosen_plots_swap_202502/barplot_swap_MABe.png')

if __name__ == '__main__':
    plot_average()