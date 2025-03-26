import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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

    files = {
        'DF3D': '/home/france/Mounted_dir/results_behavior/outputs/2023-12-06_DF3D_newnewmissing/compare_models_eval_2024-09-23/total_metrics_repeat-0.csv',
        'CLB': '/home/france/frose1_ada/results_behavior/outputs/2023-12-12_CLB_newnewmissing/compare_models_eval_2024-09-23/total_metrics_repeat-0.csv',
        'FL2': '/home/france/frose1_ada/results_behavior/outputs/2023-10-16_FL2_newnewmissing/compare_models_eval_2024-09-23/total_metrics_repeat-0.csv',
        'Human': '/home/france/Mounted_dir/results_behavior/outputs/2023-12-07_Mocap_newnewmissing/compare_models_eval_2024-09-23/total_metrics_repeat-0.csv',
        'Rat7M': '/home/france/frose1_ada/results_behavior/outputs/2023-12-05_DANNCE_newnewmissing/compare_models_eval_2024-09-23/total_metrics_repeat-0.csv',
        '2-Fish': '/home/france/Mounted_dir/results_behavior/outputs/2023-09-27_Fishv3_newnewmissing/60stride120_models_BASELINE/compare_models_eval_2024-09-23/total_metrics_repeat-0.csv',
        '2-Mice-2D': '/home/france/frose1_ada/results_behavior/outputs/2024-02-19_MABe_task1_newnewmissing/compare_models_eval_2024-09-23/total_metrics_repeat-0.csv'
    }

    df = []
    for dataset, path_ in files.items():
        print(dataset)
        small_df = pd.read_csv(path_)
        small_df = small_df.loc[small_df['method_param'] == 'type-transformer_mu_sigma-False']
        small_df.loc[:, 'Dataset'] = dataset
        if 'repeat' not in small_df.columns:
            small_df.loc[:, 'repeat'] = 0
        small_df.loc[:, 'keypoint'] = small_df['keypoint'].apply(lambda x: x.split('_x')[0] if x[-2:] == '_x' else x)
        if 'RMSE' not in small_df.columns:
            if 'dataset' in small_df.columns:
                small_df = pd.pivot(small_df, columns='metric_type', values='metric_value',
                                    index=['method', 'method_param', 'repeat', 'Dataset', 'dataset', 'keypoint', 'id_sample', 'id_hole']).reset_index()
            else:
                small_df = pd.pivot(small_df, columns='metric_type', values='metric_value',
                                    index=['method', 'method_param', 'repeat', 'Dataset', 'keypoint', 'id_sample', 'id_hole']).reset_index()
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

    dataset_order = ['DF3D', 'CLB', 'FL2', 'Human', 'Rat7M', '2-Fish', '2-Mice-2D']
    dataset2kporder = {
        'DF3D': ['all', '2', '3', '4', '7', '8', '9', '12', '13', '14', '21', '22', '23', '26', '27', '28', '31', '32',
                 '33'],
        'CLB': ['all', 'left_coord', 'right_coord', 'left_back', 'right_back', 'left_hip', 'right_hip', 'left_knee',
                'right_knee'],
        'FL2': ['all', 'left_coord', 'right_coord', 'left_back', 'right_back', 'left_hip', 'right_hip', 'left_knee',
                'right_knee'],
        'Human': ['all', 'spine_0', 'spine_1', 'spine_2', 'spine_3', 'arm1_0', 'arm1_1', 'arm1_2', 'arm1_3', 'arm2_0',
                  'arm2_1', 'arm2_2', 'arm2_3',
                  'leg1_0', 'leg1_1', 'leg1_2', 'leg1_3', 'leg2_0', 'leg2_1', 'leg2_2', 'leg2_3'],
        'Rat7M': ['all', 'ArmL', 'ArmR', 'ElbowL', 'ElbowR', 'HeadB', 'HeadF', 'HipL', 'KneeL', 'KneeR', 'ShinL',
                  'ShinR', 'ShoulderL', 'ShoulderR', 'SpineF', 'SpineM'],
        '2-Fish': ['all', 'fish1_head', 'fish1_pec', 'fish1_tail', 'fish2_head', 'fish2_pec', 'fish2_tail'],
        '2-Mice-2D': ['all', 'kp0_animal0', 'kp1_animal0', 'kp2_animal0', 'kp3_animal0', 'kp4_animal0', 'kp5_animal0',
                      'kp6_animal0',
                      'kp0_animal1', 'kp1_animal1', 'kp2_animal1', 'kp3_animal1', 'kp4_animal1', 'kp5_animal1',
                      'kp6_animal1']}

    for metric in ['RMSE', 'MPJPE']:

        fig, axes = plt.subplots(1, 7,
                                 gridspec_kw={'width_ratios': [19, 9, 9, 21, 16, 7, 15]})

        for i, d in enumerate(dataset_order):
            list_kp = np.unique(df.loc[(df['Model'] == 'DISK') * (df['Dataset'] == d), 'keypoint'].values)
            new_list_kp = ['all'] + list(list_kp[list_kp != 'all'])
            print(d, new_list_kp)
            sns.barplot(data=df.loc[(df['Model'] == 'DISK') * (df['Dataset'] == d)], x='keypoint', y=metric,
                        # hue_order=['linear interpolation', 'DISK', 'DISK proba', 'GRU', 'GRU proba', 'STS-GCN', 'ST-GCN',
                        #            'TCN'],
                        order=dataset2kporder[d],
                        palette='flare', ax=axes[i])
            axes[i].set_xticks(np.arange(len(new_list_kp)), dataset2kporder[d], rotation=90)
        f = plt.gcf()
        f.set_figwidth(18)
        f.set_figheight(6.6)
        plt.tight_layout()
        plt.savefig(f'/home/france/Dropbox/Dropbox/2021_Koeln/bogna/fig_comparison_other_methods_202502/barplot_main_results_{metric}_per_kp_202503.svg')



    files = {
        'DF3D': '/home/france/Mounted_dir/results_behavior/outputs/2023-12-06_DF3D_newnewmissing/compare_models_eval_origcoords_2024-09-23/total_metrics_repeat-0.csv',
        'CLB': '/home/france/frose1_ada/results_behavior/outputs/2023-12-12_CLB_newnewmissing/compare_models_eval_origcoord_2024-09-23/total_metrics_repeat-0.csv',
        'FL2': '/home/france/frose1_ada/results_behavior/outputs/2023-10-16_FL2_newnewmissing/compare_models_eval_origcoord_2024-09-23/total_metrics_repeat-0.csv',
        'Human': '/home/france/Mounted_dir/results_behavior/outputs/2023-12-07_Mocap_newnewmissing/compare_models_eval_origcoords_2024-09-23/total_metrics_repeat-0.csv',
        'Rat7M': '/home/france/frose1_ada/results_behavior/outputs/2023-12-05_DANNCE_newnewmissing/compare_models_eval_origcoords_2024-09-23/total_metrics_repeat-0.csv',
        '2-Fish': '/home/france/Mounted_dir/results_behavior/outputs/2023-09-27_Fishv3_newnewmissing/60stride120_models_BASELINE/compare_models_eval_origcoords_2024-09-23/total_metrics_repeat-0.csv',
        '2-Mice-2D': '/home/france/frose1_ada/results_behavior/outputs/2024-02-19_MABe_task1_newnewmissing/compare_models_eval_origcoords_2024-09-23/total_metrics_repeat-0.csv'
    }

    df = []
    for dataset, path_ in files.items():
        print(dataset)
        small_df = pd.read_csv(path_)
        small_df = small_df.loc[small_df['method_param'] == 'type-transformer_mu_sigma-False']
        small_df.loc[:, 'Dataset'] = dataset
        if 'repeat' not in small_df.columns:
            small_df.loc[:, 'repeat'] = 0
        small_df.loc[:, 'keypoint'] = small_df['keypoint'].apply(lambda x: x.split('_x')[0] if x[-2:] == '_x' else x)
        if 'RMSE' not in small_df.columns:
            if 'dataset' in small_df.columns:
                small_df = pd.pivot(small_df, columns='metric_type', values='metric_value',
                                    index=['method', 'method_param', 'repeat', 'Dataset', 'dataset', 'keypoint', 'id_sample', 'id_hole']).reset_index()
            else:
                small_df = pd.pivot(small_df, columns='metric_type', values='metric_value',
                                    index=['method', 'method_param', 'repeat', 'Dataset', 'keypoint', 'id_sample', 'id_hole']).reset_index()
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

    for metric in ['PCK@0.01']:

        fig, axes = plt.subplots(1, 7,
                                 gridspec_kw={'width_ratios': [19, 9, 9, 21, 16, 7, 15]})

        for i, d in enumerate(dataset_order):
            list_kp = np.unique(df.loc[(df['Model'] == 'DISK') * (df['Dataset'] == d), 'keypoint'].values)
            new_list_kp = ['all'] + list(list_kp[list_kp != 'all'])
            print(d, new_list_kp)
            sns.barplot(data=df.loc[(df['Model'] == 'DISK') * (df['Dataset'] == d)], x='keypoint', y=metric,
                        # hue_order=['linear interpolation', 'DISK', 'DISK proba', 'GRU', 'GRU proba', 'STS-GCN', 'ST-GCN',
                        #            'TCN'],
                        order=dataset2kporder[d],
                        palette='flare', ax=axes[i])
            axes[i].set_xticks(np.arange(len(new_list_kp)), dataset2kporder[d], rotation=90)
        f = plt.gcf()
        f.set_figwidth(18)
        f.set_figheight(6.6)
        plt.tight_layout()
        plt.savefig(f'/home/france/Dropbox/Dropbox/2021_Koeln/bogna/fig_comparison_other_methods_202502/barplot_main_results_{metric}_per_kp_202503.svg')