import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr
import numpy as np
import importlib.util

from utils import plot_save


########################################################################################################
### ERROR VS NMISSING IN MULTIKP TRAINING - FISH
########################################################################################################

enumerate_on = [[
                    'fig4_files/29-01-24_transformer_NLL_allkp_uniformproba/test',
                    0.8, '_uniform'],
                [
                    'fig4_files/29-01-24_transformer_NLL_allkp_uniformproba/test_origcoord',
                    16, '_uniform_origcoord'],
                [
                    'fig4_files/13-11-23_transformer_NLL_allkp/test_origcoord',
                    20, '_origcoord'],
                [
                    'fig4_files/13-11-23_transformer_NLL_allkp/test',
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
        f'fig4_files/RMSE_vs_lengthhole_dist_bw_fishes_cases1-3{suffix}.png')
    plt.savefig(
        f'fig4_files/RMSE_vs_lengthhole_dist_bw_fishes_cases1-3{suffix}.svg')

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
        f'fig4_files/RMSE_vs_lengthhole_dist_bw_fishes_cases2{suffix}.png')
    plt.savefig(
        f'fig4_files/RMSE_vs_lengthhole_dist_bw_fishes_cases2{suffix}.svg')


###############################################################################################
### TABLE 1FISH vs 2FISH COMPARISON


fish1_df = pd.read_csv(
    'fig4_files/compare_1vs2fish/fish1_mean_RMSE.csv')

mean_fish1 = fish1_df.groupby('method_param')['RMSE'].agg('mean').reset_index()
mean_fish1.loc[:, 'method_param'] = mean_fish1['method_param'].apply(lambda x: {'linear_interp': 'linear interpolation',
                                                                                'type-GRU_mu_sigma-False_name-Fish_v3_60stride120': 'GRU 2fish',
                                                                                'type-GRU_mu_sigma-False_name-Fish_v3_60stride120_fish1': 'GRU fish1',
                                                                                'type-GRU_mu_sigma-True_name-Fish_v3_60stride120': 'GRU proba 2fish',
                                                                                'type-GRU_mu_sigma-True_name-Fish_v3_60stride120_fish1': 'GRU proba fish1',
                                                                                'type-transformer_mu_sigma-False_name-Fish_v3_60stride120': 'DISK 2fish',
                                                                                'type-transformer_mu_sigma-False_name-Fish_v3_60stride120_fish1': 'DISK fish1',
                                                                                'type-transformer_mu_sigma-True_name-Fish_v3_60stride120': 'DISK proba 2fish',
                                                                                'type-transformer_mu_sigma-True_name-Fish_v3_60stride120_fish1': 'DISK proba fish1'}[
    x])

fish2_df = pd.read_csv(
    'fig4_files/compare_1vs2fish/fish2_mean_RMSE.csv')

mean_fish2 = fish2_df.groupby('method_param')['RMSE'].agg('mean').reset_index()
mean_fish2.loc[:, 'method_param'] = mean_fish2['method_param'].apply(lambda x: {'linear_interp': 'linear interpolation',
                                                                                'type-GRU_mu_sigma-False_name-Fish_v3_60stride120': 'GRU 2fish',
                                                                                'type-GRU_mu_sigma-False_name-Fish_v3_60stride120_fish2': 'GRU fish2',
                                                                                'type-GRU_mu_sigma-True_name-Fish_v3_60stride120': 'GRU proba 2fish',
                                                                                'type-GRU_mu_sigma-True_name-Fish_v3_60stride120_fish2': 'GRU proba fish2',
                                                                                'type-transformer_mu_sigma-False_name-Fish_v3_60stride120': 'DISK 2fish',
                                                                                'type-transformer_mu_sigma-False_name-Fish_v3_60stride120_fish2': 'DISK fish2',
                                                                                'type-transformer_mu_sigma-True_name-Fish_v3_60stride120': 'DISK proba 2fish',
                                                                                'type-transformer_mu_sigma-True_name-Fish_v3_60stride120_fish2': 'DISK proba fish2'}[
    x])

print(mean_fish1, end='\n\n')

"""
           method_param      RMSE
0  linear interpolation  0.082552
1             GRU 2fish  0.029791
2             GRU fish1  0.031453
3       GRU proba 2fish  0.030142
4       GRU proba fish1  0.032317
5            DISK 2fish  0.027404
6            DISK fish1  0.029654
7      DISK proba 2fish  0.027339
8      DISK proba fish1  0.029356
"""

print(mean_fish2)

"""
           method_param      RMSE
0  linear interpolation  0.080582
1             GRU 2fish  0.031903
2             GRU fish2  0.034525
3       GRU proba 2fish  0.032301
4       GRU proba fish2  0.034233
5            DISK 2fish  0.029530
6            DISK fish2  0.033632
7      DISK proba 2fish  0.029441
8      DISK proba fish2  0.032561
"""