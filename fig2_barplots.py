import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

###########################################################################
### BARPLOT MAIN RESULT FROm FIG 2
###########################################################################

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