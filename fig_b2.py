import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr
import numpy as np
import importlib.util

from utils import plot_save

###########################################################################
### RMSE VS LENGTH MOCAP

# outputdir = '/home/france/Mounted_dir/results_behavior/outputs/2023-09-27_Fishv3_newnewmissing/compare_models_stride120'
# outputdir = '/home/france/Mounted_dir/results_behavior/outputs/2023-12-07_Mocap_newnewmissing/compare_models'
outputdir = '/home/france/Mounted_dir/results_behavior/outputs/2023-12-05_DANNCE_newnewmissing/compare_models_transformer_GRU'
outputdir = '/home/france/Mounted_dir/results_behavior/outputs/2024-02-19_MABe_task1_newnewmissing/compare_all_goodone'
dataset = '2-Mice-2D'  # 'DANNCE'
total_rmse = pd.read_csv(os.path.join(outputdir, 'total_RMSE_repeat-0.csv'))
rename_dict = {'type-GRU_mu_sigma-False': 'GRU',
               'type-GRU_mu_sigma-True': 'GRU_NLL',
               'type-TCN_mu_sigma-False': 'TCN',
               'type-ST_GCN_mu_sigma-False': 'STGCN',
               'type-STS_GCN_mu_sigma-False': 'STSGCN',
               'type-transformer_mu_sigma-False': 'transformer_baseline',
               'type-transformer_mu_sigma-True': 'transformer_NLL',
               'linear_interp': 'linear_interp'}
total_rmse.loc[:, 'method_param'] = total_rmse['method_param'].apply(lambda x: rename_dict[x])


def lineplot_all_length():
    total_rmse.loc[:, 'length_hole'] = total_rmse.loc[:, 'length_hole'].astype('float')
    mask = (total_rmse['type_RMSE'] == '3D')
    sns.lineplot(data=total_rmse.loc[mask * (total_rmse['keypoint'] == 'all'), :], x='length_hole', y='RMSE',
                 hue='method_param',
                 hue_order=['linear_interp', 'GRU', 'GRU_NLL', 'transformer_baseline', 'transformer_NLL'],
                 palette=['gray', 'gold', 'goldenrod', 'orangered', 'firebrick', ])
    plt.tight_layout()


plot_save(lineplot_all_length,
          title=f'comparison_length_hole_all_vs_RMSE_{dataset}', only_png=False,
          outputdir=outputdir)


def lineplot_all_length():
    total_rmse.loc[:, 'length_hole'] = total_rmse.loc[:, 'length_hole'].astype('float')
    mask = (total_rmse['type_RMSE'] == '3D')
    sns.lineplot(data=total_rmse.loc[mask * (total_rmse['keypoint'] == 'all'), :], x='length_hole', y='RMSE',
                 hue='method_param',
                 hue_order=['linear_interp', 'transformer_baseline', 'transformer_NLL'],
                 palette=['gray', 'orangered', 'firebrick', ])
    plt.tight_layout()


plot_save(lineplot_all_length,
          title=f'comparison_length_hole_all_vs_RMSE_transformer_{dataset}', only_png=False,
          outputdir=outputdir)

########################################################################################################
### TRAINING TIMES SUBSAMPLING XP

csv_path = '/home/france/Documents/research_journal/behavior/training_times_subsmapling_xp_fish_202312.csv'
df = pd.read_csv('/home/france/Documents/research_journal/behavior/training_times_subsmapling_xp_fish_202312.csv')


def lineplot_training_times():
    sns.lineplot(data=df.loc[df['Batch size'] == 32, :], x='Dataset size', y='Training time (min)', hue='Model',
                 hue_order=['transformer', 'GRU'],
                 palette=['gold', 'orangered'], marker='o')
    plt.xscale('log')
    plt.xlim(0.9 * df['Dataset size'].min(), 1e5)


plot_save(lineplot_training_times,
          title=f'lineplot_training_times_fish_subsampling', only_png=False,
          outputdir='/home/france/Documents/research_journal/behavior/')

########################################################################################################
### ERROR VS NMISSING IN MULTIKP TRAINING

df = pd.read_csv(
    '/home/france/Mounted_dir/results_behavior/outputs/2024-01-09_FL2_multikp/compare_models/total_RMSE_repeat-0.csv')

nb_kp_per_hole = df[df['keypoint'] != 'all']['keypoint'].apply(lambda x: len(x.split(' ')))

df.loc[df['keypoint'] != 'all', 'nb_kp_per_hole'] = nb_kp_per_hole.values

sns.lineplot(df[df['nb_kp_per_hole'] < 8], x='nb_kp_per_hole', y='RMSE', hue='method',
             hue_order=['linear_interp', 'GRU', 'transformer'], palette=['gray', 'gold', 'orangered'])

########################################################################################################
### ERROR VS NMISSING IN MULTIKP TRAINING - FISH

enumerate_on = [[
                    '/home/france/Mounted_dir/results_behavior/outputs/2023-08-29_Fish_newnewmissing/29-01-24_transformer_NLL_allkp_uniformproba/test/',
                    0.8, '_uniform'],
                [
                    '/home/france/Mounted_dir/results_behavior/outputs/2023-08-29_Fish_newnewmissing/29-01-24_transformer_NLL_allkp_uniformproba/test_origcoord',
                    16, '_uniform_origcoord'],
                [
                    '/home/france/Mounted_dir/results_behavior/outputs/2023-08-29_Fish_newnewmissing/13-11-23_transformer_NLL_allkp_origcoord',
                    20, '_origcoord'],
                [
                    '/home/france/Mounted_dir/results_behavior/outputs/2023-08-29_Fish_newnewmissing/13-11-23_transformer_NLL_allkp_test_debug',
                    1.6, '']]

for folder, ylim, suffix in enumerate_on:
    df = pd.concat([pd.read_csv(os.path.join(folder, f'total_RMSE_repeat-{i}.csv')) for i in range(5)])

    nb_kp_per_hole = df[df['keypoint'] != 'all']['keypoint'].apply(lambda x: len(x.split(' ')))


    def count_kp(x):
        parts = x.split(' ')
        count_fish1 = len([el for el in parts if 'fish1' in el])
        count_fish2 = len([el for el in parts if 'fish2' in el])
        if count_fish1 > count_fish2:
            if count_fish2 > 0:
                return f'{count_fish1} + {count_fish2}'
            else:
                return f'{count_fish1}'
        else:
            if count_fish1 > 0:
                return f'{count_fish2} + {count_fish1}'
            else:
                return f'{count_fish2}'


    missing_scheme = df[df['keypoint'] != 'all']['keypoint'].apply(count_kp)

    df.loc[df['keypoint'] != 'all', 'missing_scheme'] = missing_scheme.values
    bins = np.linspace(0, df['dist_bw_fishes'].max(), 16)
    mid_bins = (bins[1:] + bins[:-1]) / 2
    df.loc[:, 'binned_dist_bw_fishes'] = pd.cut(df['dist_bw_fishes'], bins, labels=mid_bins)
    df['binned_dist_bw_fishes'] = df['binned_dist_bw_fishes'].astype('float')

    bins = np.linspace(0, df.loc[(df['keypoint'] != 'all'), 'length_hole'].max(), 16)
    mid_bins = (bins[1:] + bins[:-1]) / 2
    df.loc[:, 'binned_length_hole'] = pd.cut(df['length_hole'], bins, labels=mid_bins)
    df['binned_length_hole'] = df['binned_length_hole'].astype('float')

    # fig, axes = plt.subplots(1, 3, sharex='all', sharey='all', figsize=(26, 9))
    # for i, (min_, max_), in enumerate([[0, 0.4], [0.4, 0.8], [0.8, 1.2]]):
    #     mask = (df['keypoint'] != 'all') * (df['binned_dist_bw_fishes'] >= min_) *(df['binned_dist_bw_fishes'] < max_) * (df['type_RMSE'] == '3D')
    #     sns.lineplot(df[mask], x='binned_length_hole', y='RMSE', hue='missing_scheme', ax=axes[i],
    #         hue_order=['1', '1 + 1', '3', '3 + 1', '3 + 2', '3 + 3'],
    #         palette=['limegreen', 'seagreen', 'red', 'orange', 'darkorange', 'grey'])
    #     axes[i].set_title(f'dist_bw_fishes in [{min_}, {max_}]')

    sns.set_style('white')
    fig, axes = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(12, 5))
    for i, (min_, max_), in enumerate([[0, 30], [30, 60]]):
        mask = (df['keypoint'] != 'all') * (df['length_hole'] >= min_) * (df['length_hole'] < max_) * (
                    df['type_RMSE'] == '3D')
        sns.lineplot(df[mask], x='binned_dist_bw_fishes', y='RMSE', hue='missing_scheme', ax=axes[i],
                     hue_order=['1', '1 + 1', '3', '3 + 1', '3 + 2', '3 + 3'],
                     palette=['royalblue', 'darkblue', 'purple', 'deeppink', 'crimson', 'grey'])
        axes[i].set_title(f'length_hole in [{min_}, {max_}]')
        plt.ylim(0, ylim)
    plt.savefig(
        f'/home/france/Documents/research_journal/behavior/RMSE_vs_lengthhole_dist_bw_fishes_cases1-3{suffix}.png')
    plt.savefig(
        f'/home/france/Documents/research_journal/behavior/RMSE_vs_lengthhole_dist_bw_fishes_cases1-3{suffix}.svg')

    fig, axes = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(12, 5))
    for i, (min_, max_), in enumerate([[0, 30], [30, 60]]):
        mask = (df['keypoint'] != 'all') * (df['length_hole'] >= min_) * (df['length_hole'] < max_) * (
                    df['type_RMSE'] == '3D')
        sns.lineplot(df[mask], x='binned_dist_bw_fishes', y='RMSE', hue='missing_scheme', ax=axes[i],
                     hue_order=['2', '2 + 1', '2 + 2'],
                     palette=['limegreen', 'seagreen', 'darkgreen'])
        axes[i].set_title(f'length_hole in [{min_}, {max_}]')
        plt.ylim(0, ylim)
    plt.savefig(
        f'/home/france/Documents/research_journal/behavior/RMSE_vs_lengthhole_dist_bw_fishes_cases2{suffix}.png')
    plt.savefig(
        f'/home/france/Documents/research_journal/behavior/RMSE_vs_lengthhole_dist_bw_fishes_cases2{suffix}.svg')

# plt.figure()
# mask = (df['keypoint'] != 'all') * df['missing_scheme'].isin(['3 + 1', '3 + 2', '1 + 1', '2 + 1', '2 + 2']) * (df['type_RMSE'] == '3D')
# sns.lineplot(df[mask], x='binned_dist_bw_fishes', y='RMSE', hue='missing_scheme')#,
# plt.ylim(0, 1)

# mask = (df['keypoint'] != 'all') * (df['type_RMSE'] == '3D')
# df.loc[mask].groupby('missing_scheme').count()

# mask = (df['keypoint'] != 'all') * (df['type_RMSE'] == '3D')
# sns.violinplot(data=df[mask], y='length_hole', x='missing_scheme')
# sns.lineplot(df[mask], x='length_hole', y='RMSE', hue='missing_scheme')#,