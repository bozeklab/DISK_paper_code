import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

###########################################################################
### BARPLOT MAIN RESULT FROM FIG 2a + Ext. Fig. 1a&b, 2.
###########################################################################


###########################################################################
### BARPLOT RMSE and MPJPE FROM MEAN METRICS
###
### Each line in the df and each point in the barplot corresponds to one
### run of the test script for the same trained model (5 runs).
### At inference time, we create the fake gaps on the fly, therefore the
### results can vary.
###
### Using normed coordinates for these plots.
###
### RMSE plot shown in main figure 2. MPJPE in extended figure A1.
###########################################################################


files = {
    'DF3D': 'fig2a_result_files/DF3D_mean_metrics.csv',
    'CLB': 'fig2a_result_files/CLB_mean_metrics.csv',
    'FL2': 'fig2a_result_files/FL2_mean_metrics.csv',
    'Human': 'fig2a_result_files/Human_mean_metrics.csv',
    'Rat7M': 'fig2a_result_files/DANNCE_mean_metrics.csv',
    '2-Fish': 'fig2a_result_files/Fish_mean_metrics.csv',
    '2-Mice-2D': 'fig2a_result_files/MABe_mean_metrics.csv'
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
    print(f"% of improvement in terms of test {metric} between DISK and linear interpolation per dataset:\n",
          df.loc[df['Model'].isin(['linear interpolation', 'DISK'])].groupby(['dataset', 'repeat']) \
          .apply(lambda x: (x.loc[x['Model'] == 'linear interpolation', metric].values[0] - \
                            x.loc[x['Model'] == 'DISK', metric].values[0]) /
                           x.loc[x['Model'] == 'linear interpolation', metric].values[0] * 100) \
          .groupby(['dataset']).agg(['mean', 'std']),
          end='\n\n')

    plt.figure()
    ax = sns.barplot(data=df, x='Dataset', hue='Model', y=metric,
                hue_order=['linear interpolation', 'DISK', 'DISK proba', 'GRU', 'GRU proba', 'STS-GCN', 'ST-GCN',
                           'TCN'],
                palette=['gray', 'orangered', 'firebrick', 'gold', 'goldenrod', 'royalblue', 'lightskyblue',
                         'forestgreen'],
                     errorbar=('sd', 1))
    pts = np.linspace(0, np.pi * 2, 24)
    circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
    vert = np.r_[circ, circ[::-1] * .7]

    open_circle = mpl.path.Path(vert)
    mystrip = sns.stripplot(
        data=df, x='Dataset', hue='Model', y=metric,
        hue_order=['linear interpolation', 'DISK', 'DISK proba', 'GRU', 'GRU proba', 'STS-GCN', 'ST-GCN',
                   'TCN'],
        palette='dark:black', linewidth=0.01,
        dodge=True, alpha=0.6, s=9, jitter=0.25,
        ax=ax, marker=open_circle,
    )
    # mystrip.set_facecolor(None)

    f = plt.gcf()
    f.set_figwidth(18)
    f.set_figheight(6.6)
    plt.savefig(f'fig2a_result_files/barplot_main_results_{metric}.svg')


###########################################################################
### BARPLOT PCK@0.01 FROM MEAN METRICS
###
### Each line in the df and each point in the barplot corresponds to one
### run of the test script for the same trained model (5 runs).
### At inference time, we create the fake gaps on the fly, therefore the
### results can vary.
###
### Using original coordinates for this plot.
###
### PCK plot shown in extended figure A1
###########################################################################

files = {
    'DF3D': 'fig2a_result_files/DF3D_origcoords_mean_metrics.csv',
    'CLB': 'fig2a_result_files/CLB_origcoords_mean_metrics.csv',
    'FL2': 'fig2a_result_files/FL2_origcoords_mean_metrics.csv',
    'Human': 'fig2a_result_files/Human_origcoords_mean_metrics.csv',
    'Rat7M': 'fig2a_result_files/DANNCE_origcoords_mean_metrics.csv',
    '2-Fish': 'fig2a_result_files/Fish_origcoords_mean_metrics.csv',
    '2-Mice-2D': 'fig2a_result_files/MABe_origcoords_mean_metrics.csv'
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
    print(f"% of improvement in terms of test {metric} between DISK and linear interpolation per dataset:\n",
          df.loc[df['Model'].isin(['linear interpolation', 'DISK'])].groupby(['dataset', 'repeat']) \
          .apply(lambda x: (x.loc[x['Model'] == 'DISK', metric].values[0] - x.loc[x['Model'] == 'linear interpolation', metric].values[0]) * 100) \
          .groupby(['dataset']).agg(['mean', 'std']),
          end='\n\n')

    plt.figure()
    ax = sns.barplot(data=df, x='Dataset', hue='Model', y=metric,
                hue_order=['linear interpolation', 'DISK', 'DISK proba', 'GRU', 'GRU proba', 'STS-GCN', 'ST-GCN',
                           'TCN'],
                palette=['gray', 'orangered', 'firebrick', 'gold', 'goldenrod', 'royalblue', 'lightskyblue',
                         'forestgreen'],
                errorbar=('sd', 1)
                )
    sns.stripplot(
        data=df, x='Dataset', hue='Model', y=metric,
        hue_order=['linear interpolation', 'DISK', 'DISK proba', 'GRU', 'GRU proba', 'STS-GCN', 'ST-GCN',
                   'TCN'],
        palette='dark:black', linewidth=0.01,
        dodge=True, alpha=0.6, s=9, jitter=0.25,
        ax=ax, marker=open_circle,
    )
    f = plt.gcf()
    f.set_figwidth(18)
    f.set_figheight(6.6)
    plt.savefig(f'fig2a_result_files/barplot_main_results_origcoords_{metric}.svg')

    ###########################################################################
    ### BARPLOT RMSE and MPJPE FROM PER KEYPOINT METRICS
    ###
    ### Each line in the df and each point in the barplot correspond to one
    ### fake gap created at inference time. Using only one run of the test script.
    ### At inference time, we create the fake gaps on the fly, therefore the
    ### results can vary.
    ###
    ### Using normed coordinates for these plots.
    ###
    ### Plots reproduced in extended figure A2
    ###########################################################################


    files = {
        'DF3D': 'fig2a_result_files/DF3D_total_metrics_repeat-0.csv',
        'CLB': 'fig2a_result_files/CLB_total_metrics_repeat-0.csv',
        'FL2': 'fig2a_result_files/FL2_total_metrics_repeat-0.csv',
        'Human': 'fig2a_result_files/Human_total_metrics_repeat-0.csv',
        'Rat7M': 'fig2a_result_files/DANNCE_total_metrics_repeat-0.csv',
        '2-Fish': 'fig2a_result_files/Fish_total_metrics_repeat-0.csv',
        '2-Mice-2D': 'fig2a_result_files/MABe_total_metrics_repeat-0.csv'
    }

    df = []
    for dataset, path_ in files.items():
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
            list_kp, n_per_kp = np.unique(df.loc[(df['Model'] == 'DISK') * (df['Dataset'] == d), 'keypoint'].values, return_counts=True)
            new_list_kp = ['all'] + list(list_kp[list_kp != 'all'])
            print(d, list(zip(new_list_kp, n_per_kp)))
            sns.barplot(data=df.loc[(df['Model'] == 'DISK') * (df['Dataset'] == d)], x='keypoint', y=metric,
                        # hue_order=['linear interpolation', 'DISK', 'DISK proba', 'GRU', 'GRU proba', 'STS-GCN', 'ST-GCN',
                        #            'TCN'],
                        errorbar=('sd', 1),
                        order=dataset2kporder[d],
                        palette='flare', ax=axes[i])
            axes[i].set_xticks(np.arange(len(new_list_kp)), dataset2kporder[d], rotation=90)
        f = plt.gcf()
        f.set_figwidth(18)
        f.set_figheight(6.6)
        plt.tight_layout()
        plt.savefig(f'fig2a_result_files/barplot_main_results_{metric}_per_kp.svg')

    ###########################################################################
    ### BARPLOT PCK@0.01 FROM PER KEYPOINT METRICS
    ###
    ### Each line in the df and each point in the barplot correspond to one
    ### fake gap created at inference time. Using only one run of the test script.
    ### At inference time, we create the fake gaps on the fly, therefore the
    ### results can vary.
    ###
    ### Using original coordinates for this plot.
    ###
    ### Plot reproduced in extended figure A2.
    ###########################################################################

    files = {
        'DF3D': 'fig2a_result_files/DF3D_origcoords_total_metrics_repeat-0.csv',
        'CLB': 'fig2a_result_files/CLB_origcoords_total_metrics_repeat-0.csv',
        'FL2': 'fig2a_result_files/FL2_origcoords_total_metrics_repeat-0.csv',
        'Human': 'fig2a_result_files/Human_origcoords_total_metrics_repeat-0.csv',
        'Rat7M': 'fig2a_result_files/DANNCE_origcoords_total_metrics_repeat-0.csv',
        '2-Fish': 'fig2a_result_files/Fish_origcoords_total_metrics_repeat-0.csv',
        '2-Mice-2D': 'fig2a_result_files/MABe_origcoords_total_metrics_repeat-0.csv'
    }

    df = []
    for dataset, path_ in files.items():
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
            # print(d, new_list_kp)
            sns.barplot(data=df.loc[(df['Model'] == 'DISK') * (df['Dataset'] == d)], x='keypoint', y=metric,
                        # hue_order=['linear interpolation', 'DISK', 'DISK proba', 'GRU', 'GRU proba', 'STS-GCN', 'ST-GCN',
                        #            'TCN'],
                        errorbar=('sd', 1),
                        order=dataset2kporder[d],
                        palette='flare', ax=axes[i])
            axes[i].set_xticks(np.arange(len(new_list_kp)), dataset2kporder[d], rotation=90)
        f = plt.gcf()
        f.set_figwidth(18)
        f.set_figheight(6.6)
        plt.tight_layout()
        plt.savefig(f'fig2a_result_files/barplot_main_results_origcoords_{metric}_per_kp.svg')


"""
PRINTED OUTPUT

% of improvement in terms of test RMSE between DISK and linear interpolation per dataset:
                                                   mean       std
dataset                                                         
DANNCE_seq_keypoints_60_stride30_fill10      64.454095  0.681301
DF3D_keypoints_60_stride5                    44.277941  2.368590
Fish_v3_60stride60                           62.538008  0.194259
INH_CLB_keypoints_1_60_wresiduals_stride0.5  81.911476  0.429759
INH_FL2_keypoints_1_60_wresiduals_stride0.5  50.106304  1.643766
MABE_task1_60stride60                        77.222370  0.566109
Mocap_keypoints_60_stride30                  90.300525  0.289625

% of improvement in terms of test MPJPE between DISK and linear interpolation per dataset:
                                                   mean       std
dataset                                                         
DANNCE_seq_keypoints_60_stride30_fill10      66.053339  0.828786
DF3D_keypoints_60_stride5                    39.792071  2.600972
Fish_v3_60stride60                           62.715771  0.228589
INH_CLB_keypoints_1_60_wresiduals_stride0.5  81.644428  0.388980
INH_FL2_keypoints_1_60_wresiduals_stride0.5  48.185207  1.752515
MABE_task1_60stride60                        77.033658  0.622608
Mocap_keypoints_60_stride30                  90.020001  0.291014

% of improvement in terms of test PCK@0.01 between DISK and linear interpolation per dataset:
                                                   mean       std
dataset                                                         
DANNCE_seq_keypoints_60_stride30_fill10      23.494982  0.508834
DF3D_keypoints_60_stride5                     0.628429  0.250843
Fish_v3_60stride60                           11.450867  0.041713
INH_CLB_keypoints_1_60_wresiduals_stride0.5  42.022988  0.786620
INH_FL2_keypoints_1_60_wresiduals_stride0.5  12.017332  0.799417
MABE_task1_60stride60                        46.166745  0.847778
Mocap_keypoints_60_stride30                  54.452205  1.242645

DF3D [('all', 57), ('12', 54), ('13', 47), ('14', 60), ('2', 58), ('21', 62), ('22', 47), ('23', 51), ('26', 59), ('27', 59), ('28', 49), ('3', 44), ('31', 57), ('32', 49), ('33', 67), ('4', 55), ('7', 52), ('8', 52), ('9', 672)]
CLB [('all', 692), ('left_back', 70), ('left_coord', 282), ('left_hip', 302), ('left_knee', 225), ('right_back', 186), ('right_coord', 191), ('right_hip', 104), ('right_knee', 76)]
FL2 [('all', 413), ('left_back', 158), ('left_coord', 74), ('left_hip', 59), ('left_knee', 107), ('right_back', 146), ('right_coord', 76), ('right_hip', 56), ('right_knee', 99)]
Human [('all', 869), ('arm1_0', 66), ('arm1_1', 69), ('arm1_2', 63), ('arm1_3', 56), ('arm2_0', 73), ('arm2_1', 58), ('arm2_2', 60), ('arm2_3', 65), ('leg1_0', 66), ('leg1_1', 71), ('leg1_2', 45), ('leg1_3', 57), ('leg2_0', 67), ('leg2_1', 64), ('leg2_2', 53), ('leg2_3', 70), ('spine_0', 67), ('spine_1', 68), ('spine_2', 64), ('spine_3', 71)]
Rat7M [('all', 172), ('ArmL', 578), ('ArmR', 149), ('ElbowL', 1027), ('ElbowR', 2), ('HeadB', 10), ('HeadF', 31), ('HipL', 220), ('KneeL', 15), ('KneeR', 235), ('ShinL', 277), ('ShinR', 89), ('ShoulderL', 886), ('ShoulderR', 115), ('SpineF', 84), ('SpineM', 2713)]
2-Fish [('all', 15705), ('fish1_head', 4021), ('fish1_pec', 4463), ('fish1_tail', 5385), ('fish2_head', 4660), ('fish2_pec', 4726), ('fish2_tail', 6042)]
2-Mice-2D [('all', 622), ('kp0_animal0', 58), ('kp0_animal1', 61), ('kp1_animal0', 71), ('kp1_animal1', 54), ('kp2_animal0', 68), ('kp2_animal1', 73), ('kp3_animal0', 61), ('kp3_animal1', 62), ('kp4_animal0', 72), ('kp4_animal1', 61), ('kp5_animal0', 79), ('kp5_animal1', 75), ('kp6_animal0', 63), ('kp6_animal1', 66)]
"""