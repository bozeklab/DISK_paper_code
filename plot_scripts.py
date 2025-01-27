import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr
import numpy as np
import importlib.util

from utils import plot_save


###########################################################################
### N MISSING PLOT

df = pd.read_csv('/home/france/Dropbox/Dropbox/2021_Koeln/behavior/fl2_n_missing_results.csv')
sns.lineplot(df, x='n_missing', hue='method', y='test error',
             hue_order=['linear_interp', 'GRU', 'transformer_baseline'], palette=['gray', 'gold', 'orangered'])

###########################################################################
### BARPLOT MAIN RESULT

# files = {'DF3D': '/home/france/Mounted_dir/results_behavior/outputs/2023-12-06_DF3D_newnewmissing/compare_all_models_with0_eval/mean_RMSE.csv',
#          'CLB': '/home/france/Mounted_dir/results_behavior/outputs/2023-12-12_CLB_newnewmissing/compare_models_new_eval_test/mean_RMSE.csv',
#          'FL2': '/home/france/Mounted_dir/results_behavior/outputs/2023-10-16_FL2_newnewmissing/compare_models_new_eval/mean_RMSE.csv',
#          'Human': '/home/france/Mounted_dir/results_behavior/outputs/2023-12-07_Mocap_newnewmissing/compare_models_new_eval/mean_RMSE.csv',
#          'Rat7M': '/home/france/Mounted_dir/results_behavior/outputs/2023-12-05_DANNCE_newnewmissing/compare_models_new_eval/mean_RMSE.csv',
#          '2-Fish': '/home/france/Mounted_dir/results_behavior/outputs/2023-09-27_Fishv3_newnewmissing/60stride120_models_BASELINE/compare_modelsnew_eval/mean_RMSE.csv',
#          '2-Mice-2D': '/home/france/Mounted_dir/results_behavior/outputs/2024-02-19_MABe_task1_newnewmissing/compare_all_goodone/mean_RMSE.csv'
#          }

files = {
    'DF3D': '/home/france/Mounted_dir/results_behavior/outputs/2023-12-06_DF3D_newnewmissing/compare_models_eval_2024-09-23/mean_metrics.csv',
    'CLB': '/home/france/frose1_ada/results_behavior/outputs/2023-12-12_CLB_newnewmissing/compare_models_eval_2024-09-23/mean_metrics.csv',
    'FL2': '/home/france/frose1_ada/results_behavior/outputs/2023-10-16_FL2_newnewmissing/compare_models_eval_2024-09-23/mean_metrics.csv',
    'Human': '/home/france/Mounted_dir/results_behavior/outputs/2023-12-07_Mocap_newnewmissing/compare_models_eval_2024-09-23/mean_metrics.csv',
    'Rat7M': '/home/france/frose1_ada/results_behavior/outputs/2023-12-05_DANNCE_newnewmissing/compare_models_eval_2024-09-23/mean_metrics.csv',
    '2-Fish': '/home/france/Mounted_dir/results_behavior/outputs/2023-09-27_Fishv3_newnewmissing/60stride120_models_BASELINE/compare_models_eval_2024-09-23/mean_metrics.csv',
    '2-Mice-2D': '/home/france/frose1_ada/results_behavior/outputs/2024-02-19_MABe_task1_newnewmissing/compare_models_eval_2024-09-23/mean_metrics.csv'
    }

df = []
for dataset, path_ in files.items():
    small_df = pd.read_csv(path_)
    small_df.loc[:, 'Dataset'] = dataset
    if 'RMSE' not in small_df.columns:
        small_df = pd.pivot(small_df, columns='metric_type', values='metric_value',
                            index=['method', 'method_param', 'repeat', 'Dataset', 'dataset']).reset_index()
    df.append(small_df)
df = pd.concat(df)
dict_ = {'type-GRU_mu_sigma-False': 'GRU',
         'type-GRU_mu_sigma-True': 'GRU proba',
         'type-STS_GCN_mu_sigma-False': 'STS-GCN',
         'type-ST_GCN_mu_sigma-False': 'ST-GCN',
         'type-TCN_mu_sigma-False': 'TCN',
         'linear_interp': 'linear interpolation',
         'type-transformer_mu_sigma-False': 'DISK',
         'type-transformer_mu_sigma-True': 'DISK proba'
         }
df['Model'] = df['method_param'].apply(lambda x: dict_[x])

for metric in ['RMSE', 'MPJPE']:
    print("% of improvement in terms of test RMSE between DISK and linear interpolation per dataset:\n",
          df.loc[df['Model'].isin(['linear interpolation', 'DISK'])].groupby(['dataset', 'repeat']) \
          .apply(lambda x: (x.loc[x['Model'] == 'linear interpolation', metric].values[0] - \
                            x.loc[x['Model'] == 'DISK', 'RMSE'].values[0]) /
                           x.loc[x['Model'] == 'linear interpolation', metric].values[0] * 100) \
          .groupby(['dataset']).agg(['mean', 'std']))

    plt.figure()
    sns.barplot(data=df, x='Dataset', hue='Model', y=metric,
                hue_order=['linear interpolation', 'DISK', 'DISK proba', 'GRU', 'GRU proba', 'STS-GCN', 'ST-GCN',
                           'TCN'],
                palette=['gray', 'orangered', 'firebrick', 'gold', 'goldenrod', 'royalblue', 'lightskyblue',
                         'forestgreen'])
    f = plt.gcf()
    f.set_figwidth(18)
    f.set_figheight(6.6)
    plt.savefig(f'/home/france/Dropbox/Dropbox/2021_Koeln/bogna/barplot_main_results_{metric}_202409.svg')

files = {
    'DF3D': '/home/france/Mounted_dir/results_behavior/outputs/2023-12-06_DF3D_newnewmissing/compare_models_eval_origcoords_2024-09-23/mean_metrics.csv',
    'CLB': '/home/france/frose1_ada/results_behavior/outputs/2023-12-12_CLB_newnewmissing/compare_models_eval_origcoord_2024-09-23/mean_metrics.csv',
    'FL2': '/home/france/frose1_ada/results_behavior/outputs/2023-10-16_FL2_newnewmissing/compare_models_eval_origcoord_2024-09-23/mean_metrics.csv',
    'Human': '/home/france/Mounted_dir/results_behavior/outputs/2023-12-07_Mocap_newnewmissing/compare_models_eval_origcoords_2024-09-23/mean_metrics.csv',
    'Rat7M': '/home/france/frose1_ada/results_behavior/outputs/2023-12-05_DANNCE_newnewmissing/compare_models_eval_origcoords_2024-09-23/mean_metrics.csv',
    '2-Fish': '/home/france/Mounted_dir/results_behavior/outputs/2023-09-27_Fishv3_newnewmissing/60stride120_models_BASELINE/compare_models_eval_origcoords_2024-09-23/mean_metrics.csv',
    '2-Mice-2D': '/home/france/frose1_ada/results_behavior/outputs/2024-02-19_MABe_task1_newnewmissing/compare_models_eval_origcoords_2024-09-23/mean_metrics.csv'
    }

df = []
for dataset, path_ in files.items():
    small_df = pd.read_csv(path_)
    small_df.loc[:, 'Dataset'] = dataset
    if 'RMSE' not in small_df.columns:
        small_df = pd.pivot(small_df, columns='metric_type', values='metric_value',
                            index=['method', 'method_param', 'repeat', 'Dataset', 'dataset']).reset_index()
    df.append(small_df)
df = pd.concat(df)

df['Model'] = df['method_param'].apply(lambda x: dict_[x])

for metric in ['PCK@0.01']:
    print("% of improvement in terms of test RMSE between DISK and linear interpolation per dataset:\n",
          df.loc[df['Model'].isin(['linear interpolation', 'DISK'])].groupby(['dataset', 'repeat']) \
          .apply(lambda x: (x.loc[x['Model'] == 'linear interpolation', metric].values[0] - \
                            x.loc[x['Model'] == 'DISK', 'RMSE'].values[0]) /
                           x.loc[x['Model'] == 'linear interpolation', metric].values[0] * 100) \
          .groupby(['dataset']).agg(['mean', 'std']))

    plt.figure()
    sns.barplot(data=df, x='Dataset', hue='Model', y=metric,
                hue_order=['linear interpolation', 'DISK', 'DISK proba', 'GRU', 'GRU proba', 'STS-GCN', 'ST-GCN',
                           'TCN'],
                palette=['gray', 'orangered', 'firebrick', 'gold', 'goldenrod', 'royalblue', 'lightskyblue',
                         'forestgreen'])
    f = plt.gcf()
    f.set_figwidth(18)
    f.set_figheight(6.6)
    plt.savefig(f'/home/france/Dropbox/Dropbox/2021_Koeln/bogna/barplot_main_results_origcoords_{metric}_202409.svg')

    # datasets = df['Dataset'].unique()
    # n_datasets = len(datasets)
    # fig, axes = plt.subplots(1, n_datasets)
    # for i in range(n_datasets):
    #     sns.barplot(data=df[df['Dataset']==datasets[i]], x='Dataset', hue='Model', y=f'test_{metric}_orig_coord',
    #                 hue_order=['linear_interp', 'transformer_baseline', 'transformer_NLL', 'GRU', 'GRU_NLL', 'TCN',  'STSGCN', 'STGCN'],
    #                 palette=['gray', 'orangered', 'firebrick', 'gold', 'goldenrod', 'forestgreen', 'royalblue', 'lightskyblue'],
    #                 ax=axes[i])
    # fig.set_figwidth(18)
    # fig.set_figheight(6)
    # plt.tight_layout()

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


###############################################################################################
### REDO CORR PLOT


df = pd.read_csv(
    '/home/france/Mounted_dir/results_behavior/outputs/2023-09-27_Fishv3_newnewmissing/60stride120_models_BASELINE/compare_modelsnew_eval/total_RMSE_repeat-0.csv')

pivot_df = pd.pivot(
    df.loc[(df['keypoint'] == 'all') * (df['method_param'] == 'type-transformer_mu_sigma-True'), :],
    values='RMSE', index='id_sample', columns='type_RMSE')
pivot_df['mean_uncertainty'] = pivot_df['mean_uncertainty'].astype(float)
pivot_df['3D'] = pivot_df['3D'].astype(float)
pcoeff, ppval = pearsonr(pivot_df['3D'].values, pivot_df['mean_uncertainty'].values)
print(f'Model type-transformer_mu_sigma-True: PEARSONR COEFF {pcoeff}, PVAL {ppval}')


def corr_plot():
    plt.figure()
    ax = sns.histplot(data=pivot_df, x='3D', y='mean_uncertainty', bins=70, color='midnightblue')
    for violin in ax.collections[::2]:
        violin.set_alpha(1)
    sns.kdeplot(data=pivot_df, x='3D', y='mean_uncertainty')
    plt.plot([0, pivot_df['3D'].max()], [0, pivot_df['3D'].max()], 'r--')
    plt.title(f'Pearson coeff: {pcoeff:.3f}')


sns.set_style('darkgrid')
corr_plot()

###############################################################################################
### TABLE 1FISH vs 2FISH COMPARISON


fish1_df = pd.read_csv(
    '/home/france/Mounted_dir/results_behavior/outputs/2024-02-21_Fish_v3_singlefish_newnewmissing/compare_1fish_2fish_testfish1/mean_RMSE.csv')

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
    '/home/france/Mounted_dir/results_behavior/outputs/2024-02-21_Fish_v3_singlefish_newnewmissing/compare_all_fish2/mean_RMSE.csv')

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


##################################################################################################
### ERROR BY ACTION / DIST

def binning(x):
    if x < 10:
        return 0
    elif x < 20:
        return 1
    elif x < 30:
        return 2
    elif x < 40:
        return 3
    elif x < 50:
        return 4
    else:
        return 5


def binning_dist(x):
    if x < 0.5:
        return 0
    elif x < 1:
        return 1
    elif x < 1.5:
        return 2
    elif x < 2:
        return 3
    elif x < 2.5:
        return 4
    else:
        return 5


## MABe
df = pd.read_csv(
    '/home/france/Mounted_dir/results_behavior/models/MABe_10-44-22_transformer_NLL/MABE_task1_60stride60_metadata_swapped_PCA_proba0.1.csv')
df.loc[:, 'dist_bw_mice_bin'] = df['dist_bw_mice'].apply(binning_dist)
df.loc[:, 'n_missing_bin'] = df['n_missing'].apply(binning)

plt.figure('action_vs_rmse')
sns.barplot(data=df[df['swapped'] == False], x='action_str', y='rmse')
plt.savefig('/home/france/Dropbox/Dropbox/2021_Koeln/bogna/barplot_MABe_rmse_vs_action.svg')
plt.savefig('/home/france/Dropbox/Dropbox/2021_Koeln/bogna/barplot_MABe_rmse_vs_action.png')

plt.figure('action_vs_rmse_high_n_missing_sup40')
sns.barplot(data=df[(df['swapped'] == False) * (df['n_missing_bin'] > 4)], x='action_str', y='rmse')
plt.savefig('/home/france/Dropbox/Dropbox/2021_Koeln/bogna/barplot_MABe_rmse_vs_action_high_n_missing_sup40.svg')
plt.savefig('/home/france/Dropbox/Dropbox/2021_Koeln/bogna/barplot_MABe_rmse_vs_action_high_n_missing_sup40.png')

# plt.figure('action_vs_dist_bw_mice')
# sns.lineplot(data=df[df['swapped']==False], x='dist_bw_mice_bin', hue='action_str', y='rmse')

# plt.figure('action_vs_n_missing')
# sns.lineplot(data=df[df['swapped']==False], x='n_missing_bin', hue='action_str', y='rmse')

# plt.figure('action_vs_n_missing_high_dist')
# sns.lineplot(data=df[(df['swapped']==False) * (df['dist_bw_mice_bin'] > 3)], x='n_missing_bin', hue='action_str', y='rmse')

# plt.figure('action_vs_n_missing_low_dist')
# sns.lineplot(data=df[(df['swapped']==False) * (df['dist_bw_mice_bin'] < 1)], x='n_missing_bin', hue='action_str', y='rmse')

df = pd.read_csv(
    '/home/france/Mounted_dir/results_behavior/models/03-10-24_transformer_NLL/Mocap_keypoints_60_stride30_swapped_PCA_proba0.1.csv')
df.loc[:, 'n_missing_bin'] = df['n_missing'].apply(binning)

plt.figure('action_vs_rmse')
sns.barplot(data=df[df['swapped'] == False], x='action_str', y='rmse')
plt.savefig('/home/france/Dropbox/Dropbox/2021_Koeln/bogna/barplot_human_rmse_vs_action.svg')
plt.savefig('/home/france/Dropbox/Dropbox/2021_Koeln/bogna/barplot_human_rmse_vs_action.png')

plt.figure('action_vs_rmse_high_n_missing')
sns.barplot(data=df[(df['swapped'] == False) * (df['n_missing_bin'] > 4)], x='action_str', y='rmse')
plt.savefig('/home/france/Dropbox/Dropbox/2021_Koeln/bogna/barplot_human_rmse_vs_action_high_n_missing_sup40.svg')
plt.savefig('/home/france/Dropbox/Dropbox/2021_Koeln/bogna/barplot_human_rmse_vs_action_high_n_missing_sup40.png')

# plt.figure('action_vs_n_missing')
# sns.lineplot(data=df[df['swapped']==False], x='n_missing_bin', hue='action_str', y='rmse')


#####################################################################################################
### PLOTS WITH GREATER CONTEXT WINDOWS (FROM IMPUTED VS NON-IMPUTED FULLLENGTH NPZ FILES)

orig = np.load(
    '/home/france/Mounted_dir/results_behavior/datasets/INHCP_CLB_keypoints_60stride30/val_fulllength_dataset_w-all-nans.npz')
imputed = np.load(
    '/home/france/Mounted_dir/results_behavior/datasets/INHCP_CLB_keypoints_60stride30/val_fulllength_dataset_imputed.npz')
dataset_constants = read_constant_file(
    '/home/france/Mounted_dir/results_behavior/datasets/INHCP_CLB_keypoints_60stride30/constants.py')
X_orig = orig['X']
X_imputed = imputed['X']

# for i_file in range(13):

i_file = 11
fig, axes = plt.subplots(8, 3, figsize=(30, 8), sharex='all', sharey='none')
time = np.arange(X_imputed.shape[1]) / 60
for i in range(8):
    for j in range(3):
        axes[i, j].plot(time, X_imputed[i_file, :, 3 * i + j], 'ro', ms=1)
        axes[i, j].plot(time, X_orig[i_file, :, 3 * i + j], 'bo', ms=1)
        axes[i, j].set_xlim(-0.5, 22)
    axes[i, 0].set_ylabel(dataset_constants.KEYPOINTS[i])
axes[7, 1].set_xlabel('Time(sec)')
axes[0, 0].set_title('X')
axes[0, 1].set_title('Y')
axes[0, 2].set_title('Z')
plt.tight_layout()
plt.savefig(f'/home/france/Dropbox/Dropbox/2021_Koeln/bogna/longer_plot_impute_INHCP_CLB_60stride30_file{i_file}.svg')
plt.savefig(f'/home/france/Dropbox/Dropbox/2021_Koeln/bogna/longer_plot_impute_INHCP_CLB_60stride30_file{i_file}.png')

#######################################################################################################
### PROPORTION OF EACH CLASS MABE


mabe_traidata = np.array(np.load('/home/france/Mounted_dir/behavior_data/MABe_task1/train.npy', allow_pickle=True))
mbe_train = mabe_traidata.item()
seq_names = ['1cac195d39', 'b40d39ca72', 'f45694e6b9', '9212f26324', '5490af36aa', '4ce1975da0', 'b62aff1dc4',
             'df1728596d', 'f06c359e8f', '527dbf1279', '1fed911c1c', '9067e84eb9', '12f1d0d6ff', 'eac63ca45e',
             '70fbc2c277', '9160352ddd', '3b9324f65c', 'e2cd67e1a3', '8f6193714b', 'f34ed34319', '0d7c7eb4b2',
             'fa86bc4dc8', 'b2ed66d49b', 'ecdd2509b3', '2070ae2860', '18cd199663', '4102dd1c7e', 'd1d1da0f45',
             '24d04a0320', 'ed0e470aea', '0b278ffe3d', '61caaf5764', '4aa2e49a0b', '30b97085a2', '8331b66dc4',
             'e3b5115e7b', '0f7327e2c8', 'e2d0f167da', '22f1628272', '736d6f564a', '677b9abbc8', '1387d17de0',
             'fcfcb9c243', '6b1df151b9', '6b6b0b3ead', '83bf0c4244', '283a94cc8e', '00db873de6', '58e6d37468',
             'b0be325515', 'a908230f0e', '98f42f62ec', '7053f4a2d0', 'e85486f7ea', 'fc572dab52', '5587e73462',
             'a4c03eaf9d', '8c04221537', '97b56c159b', '3ad427f000', 'bb97f42a1b', '747858dc95', 'ecbeca2d0e',
             'af86182e32', '99e98c7b00', 'bf702e3c78', '0f38732507', 'f5c7687579', 'cd5fbeaa98', 'ff478456a9']

counts = np.array([0, 0, 0, 0])
for s in seq_names:
    annot = mbe_train['sequences'][s]['annotations']
    uniq, c = np.unique(annot, return_counts=True)
    counts[uniq] += c

print(mbe_train['vocabulary'], counts / np.sum(counts))
# per timeframe
# {'attack': 0, 'investigation': 1, 'mount': 2, 'other': 3}
# [ 14043, 146623,  28622, 318450]
# [0.02765797, 0.28877689, 0.05637159, 0.62719355]

### for human dataset

df_human = pd.read_csv('/home/france/Mounted_dir/behavior_data/mocap_dataset/pose_clip.csv')
npy_files = glob('/home/france/Mounted_dir/behavior_data/mocap_dataset/mocap_3djoints/*.npy')
npy_names = [os.path.basename(f).split('.')[0] for f in npy_files]
for n in npy_names:
    if len(df_human.loc[df_human['motion'] == n, :]) == 0:
        print(n)
        df_human.loc[df_human.shape[0], :] = [n, 'Unknown']
cat, counts = np.unique(df_human['action_type'], return_counts=True)
print(cat, counts / np.sum(counts))

# per sequence (one file = one label)
# ['Animal Behavior', 'Climb', 'Dance', 'Jump', 'Run', 'Step', 'Walk', 'Wash']
# [101,  33,  77, 108, 109,  68, 474, 118]
# [0.09283088, 0.03033088, 0.07077206, 0.09926471, 0.10018382, 0.0625    , 0.43566176, 0.10845588]

# ['Animal Behavior' 'Climb'    'Dance'    'Jump'     'Run'     'Step'      'Unknown'  'Walk'     'Wash']
# [0.04017502        0.01312649 0.03062848 0.04295943 0.0433572  0.02704853 0.56722355 0.18854415 0.04693715]


###################################################################################################
###

df_mean = pd.read_csv(
    '/home/france/Mounted_dir/results_behavior/outputs/test_compare_MABe_SWAP_origcoord/mean_metrics.csv')

renamemethods_dict = {'linear_interp': 'linear interpolation',
                      'mask-False_swap-0.1_2': 'DISK Swap woMask',
                      'mask-False_swap-0_0': 'DISK woSwap woMask',
                      'mask-False_swap-0_4': '',
                      'mask-True_swap-0.1_1': 'DISK Swap Mask',
                      'mask-True_swap-0_3': 'DISK woSwap Mask'}

df_mean.loc[:, 'method'] = df_mean['method_param'].apply(lambda x: renamemethods_dict[x])

fig, axes = plt.subplots(1, 3, sharey='col', figsize=(12, 6))
sns.barplot(data=df_mean.loc[df_mean['metric_type'] == 'RMSE'], x='metric_type', y='metric_value', hue='method',
            hue_order=['linear interpolation', 'DISK woSwap woMask', 'DISK woSwap Mask', 'DISK Swap woMask',
                       'DISK Swap Mask'],
            palette=['grey', 'green', 'gold', 'orangered', 'purple'], ax=axes[0])
sns.barplot(data=df_mean.loc[df_mean['metric_type'] == 'PCK@0.01'], x='metric_type', y='metric_value', hue='method',
            hue_order=['linear interpolation', 'DISK woSwap woMask', 'DISK woSwap Mask', 'DISK Swap woMask',
                       'DISK Swap Mask'],
            palette=['grey', 'green', 'gold', 'orangered', 'purple'], ax=axes[1])
axes[1].set_ylim(0, 1.)
plt.legend([])
sns.barplot(data=df_mean.loc[df_mean['metric_type'] == 'MPJPE'], x='metric_type', y='metric_value', hue='method',
            hue_order=['linear interpolation', 'DISK woSwap woMask', 'DISK woSwap Mask', 'DISK Swap woMask',
                       'DISK Swap Mask'],
            palette=['grey', 'green', 'gold', 'orangered', 'purple'], ax=axes[2])
plt.legend([])
plt.savefig('/home/france/Dropbox/Dropbox/2021_Koeln/bogna/chosen_plots_swap_202409/barplot_swap_MABe_origccords.svg')
plt.savefig('/home/france/Dropbox/Dropbox/2021_Koeln/bogna/chosen_plots_swap_202409/barplot_swap_MABe_origccords.png')

df_mean = pd.read_csv('/home/france/Mounted_dir/results_behavior/outputs/test_compare_MABe_SWAP/mean_metrics.csv')
df_mean.loc[:, 'method'] = df_mean['method_param'].apply(lambda x: renamemethods_dict[x])

fig, axes = plt.subplots(1, 3, sharey='col', figsize=(12, 6))
sns.barplot(data=df_mean.loc[df_mean['metric_type'] == 'RMSE'], x='metric_type', y='metric_value', hue='method',
            hue_order=['linear interpolation', 'DISK woSwap woMask', 'DISK woSwap Mask', 'DISK Swap woMask',
                       'DISK Swap Mask'],
            palette=['grey', 'forestgreen', 'gold', 'red', 'purple'], ax=axes[0])
sns.barplot(data=df_mean.loc[df_mean['metric_type'] == 'PCK@0.01'], x='metric_type', y='metric_value', hue='method',
            hue_order=['linear interpolation', 'DISK woSwap woMask', 'DISK woSwap Mask', 'DISK Swap woMask',
                       'DISK Swap Mask'],
            palette=['grey', 'forestgreen', 'gold', 'red', 'purple'], ax=axes[1])
axes[1].set_ylim(0, 1.)
plt.legend([])
sns.barplot(data=df_mean.loc[df_mean['metric_type'] == 'MPJPE'], x='metric_type', y='metric_value', hue='method',
            hue_order=['linear interpolation', 'DISK woSwap woMask', 'DISK woSwap Mask', 'DISK Swap woMask',
                       'DISK Swap Mask'],
            palette=['grey', 'forestgreen', 'gold', 'red', 'purple'], ax=axes[2])
plt.legend([])
plt.savefig('/home/france/Dropbox/Dropbox/2021_Koeln/bogna/chosen_plots_swap_202409/barplot_swap_MABe.svg')
plt.savefig('/home/france/Dropbox/Dropbox/2021_Koeln/bogna/chosen_plots_swap_202409/barplot_swap_MABe.png')

##################################################################################################################
### TABLE PAD TEST FL2

df = []
for i in [0, 1, 2, 5, 10]:
    for j in [0, 1, 2, 5, 10]:
        folder = f'/home/france/Mounted_dir/results_behavior/outputs/2023-08-29_FL2_newmissing/test_pad-{i}-{j}'
        tmp = pd.read_csv(os.path.join(folder, 'mean_metrics.csv'))
        if i == 0 or j == 0:
            tmp.loc[tmp.shape[0], :] = ['RMSE', 'linear_interp', 'linear_interp', np.nan, 0, '']
        tmp.loc[:, 'pad before'] = i
        tmp.loc[:, 'pad after'] = j

        df.append(tmp)
df = pd.concat(df)

square_linear = \
df.loc[(df['method'] == 'linear_interp') * (df['metric_type'] == 'RMSE')].groupby(['pad before', 'pad after'])[
    'metric_value'].mean().reset_index().pivot(index='pad before', columns='pad after', values='metric_value')
square_DISK = \
df.loc[(df['method'] == 'transformer') * (df['metric_type'] == 'RMSE')].groupby(['pad before', 'pad after'])[
    'metric_value'].mean().reset_index().pivot(index='pad before', columns='pad after', values='metric_value')

fig, axes = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(14, 5))
sns.heatmap(data=square_linear, annot=True, vmin=0.022, vmax=0.071, ax=axes[0], cmap='Reds')
sns.heatmap(data=square_DISK, annot=True, vmin=0.022, vmax=0.071, ax=axes[1], cmap='Reds')
axes[0].set_title('linear interpolation')
axes[1].set_title('DISK')
axes[0].set_ylabel('pad before')
plt.tight_layout()
plt.savefig('/home/france/Dropbox/Dropbox/2021_Koeln/bogna/heatmaps_FL2_pad_tests.svg')
plt.savefig('/home/france/Dropbox/Dropbox/2021_Koeln/bogna/heatmaps_FL2_pad_tests.png')