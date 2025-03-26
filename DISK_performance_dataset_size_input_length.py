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

