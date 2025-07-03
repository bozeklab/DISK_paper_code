
import os, sys
import tqdm
from glob import glob

import numpy as np
import pandas as pd
from skimage.io import imread, imsave
from statannotations.Annotator import Annotator
from scipy.stats import chi2_contingency

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

dict_label2int = {'INH1A': {'experiment': {'INH1A': 0},
                            'treatment': {'A': 0, 'B': 1, 'C': 2},
                            'treatment_detail': {'A': 0, 'B': 1.5, 'C': 2.5},
                            'mouse_id': {'M1': 0, 'M2': 1, 'M3': 2, 'M4': 3, 'M5': 4,
                                         'M6': 5, 'M7': 6, 'M8': 7, 'M9': 8, 'M10': 9},
                            'experiment_type': {'FL2': 0, 'CLB': 1}
                            },
                  'INH2B': {'experiment': {'INH2B': 1},
                            'treatment': {'A': 0, 'B': 3, 'C': 1, 'D': 2, 'E': 4, 'F': 5},
                            'treatment_detail': {'A': 0, 'B': 3, 'C': 1, 'D': 2, 'E': 4, 'F': 5},
                            'mouse_id': {'M1': 10, 'M2': 11, 'M3': 12, 'M4': 13, 'M5': 14, 'M6': 15,
                                         'M7': 16, 'M8': 17, 'M9': 18, 'M10': 19, 'M11': 20, 'M12': 21},
                            'experiment_type': {'FL2': 0, 'CLB': 1}
                            },
                  'CP1': {'experiment': {'CP1': 2},
                           'treatment': {'A': 0, 'B': 6, 'C': 6, 'D': 6},
                           'treatment_detail': {'A': 0, 'B': 6, 'C': 6.3, 'D': 6.6},
                           'mouse_id': {'M1': 22, 'M14': 23, 'M15': 24, 'M19': 25},
                           'experiment_type': {'FL2': 0, 'CLB': 1}
                           },
                  'CP1B': {'experiment': {'CP1B': 3},
                           'treatment': {'A': 0, 'B': 6, 'C': 6, 'D': 6},
                           'treatment_detail': {'A': 0, 'B': 6, 'C': 6.3, 'D': 6.6},
                           'mouse_id': {'M1': 26, 'M2': 27, 'M3': 28, 'M4': 29, 'M5': 30, 'M6': 31},
                           'experiment_type': {'FL2': 0, 'CLB': 1}
                           },
                  }

def get10sec_window(x):
    freq = 60
    if x <= freq * 10:
        return 0
    elif x <= freq * 20:
        return 1
    elif x <= freq * 30:
        return 2
    elif x <= freq * 40:
        return 3
    elif x <= freq * 50:
        return 4
    else:
        return 5


if __name__ == '__main__':

    basedir = '../datasets/INHCP_FL2_keypoints_60stride30/'
    featuresdir = os.path.join(basedir, 'step_features/')
    threshold_count_steps = 10

    keypoints = ['left_hip', 'right_hip', 'left_back', 'right_back', 'left_knee', 'left_ankle', 'right_knee', 'right_ankle']
    data_imputed = np.load(os.path.join(basedir, 'train_fulllength_dataset_imputed.npz'))
    data_before = np.load(os.path.join(basedir, 'train_fulllength_dataset_w-all-nans.npz'))
    X_imputed = data_imputed['X']
    X_imputed = X_imputed.reshape(X_imputed.shape[0], X_imputed.shape[1], len(keypoints), -1)[..., 0]
    X_before = data_before['X']
    X_before = X_before.reshape(X_before.shape[0], X_before.shape[1], len(keypoints), -1)[..., 0]
    percent_nan_imputed = np.array([np.sum(np.isnan(x), axis=0) / X_imputed.shape[1] * 100 for x in X_imputed]).flatten()
    percent_nan_before = np.array([np.sum(np.isnan(x), axis=0) / X_before.shape[1] * 100 for x in X_before]).flatten()
    percent_nan_per_kp_df = pd.DataFrame(columns=['keypoint', 'percent_nan', 'index_sample'],
                                  data=np.vstack([np.tile(keypoints, X_imputed.shape[0] + X_before.shape[0]),
                                                  np.concatenate([percent_nan_before, percent_nan_imputed]),
                                                  np.repeat(list(np.arange(X_imputed.shape[0])) + list(np.arange(X_before.shape[0])), len(keypoints))]).T)
    percent_nan_per_kp_df.loc[:X_imputed.shape[0] * len(keypoints), 'imputed'] = False
    percent_nan_per_kp_df.loc[X_imputed.shape[0] * len(keypoints):, 'imputed'] = True
    percent_nan_per_kp_df.loc[:, 'percent_nan'] = percent_nan_per_kp_df['percent_nan'].astype('float')
    percent_nan_per_kp_df.loc[:, 'index_sample'] = percent_nan_per_kp_df['index_sample'].astype('int')

    df_before = pd.read_csv(os.path.join(featuresdir, f'swing_properties_swing_duration_train_fulllength.csv'))
    df_after = pd.read_csv(os.path.join(featuresdir, f'swing_properties_swing_duration_train_fulllength_imputed.csv'))

    df_after.loc[:, 'imputed'] = True
    df_before.loc[:, 'imputed'] = False
    df = pd.concat([df_before, df_after])
    df.loc[:, 'side_imputed'] = df[['side', 'imputed']].apply(lambda x: f'{x[0]}_{["orig", "imputed"][x[1]]}', axis=1)
    df = df.reset_index()
    df.loc[:, '10second_window'] = df['vel_peak_index'].apply(get10sec_window)
    df.loc[:, 'window_10sec'] = df['10second_window'].astype(str)
    print(df.groupby(['treatment_str', 'side', 'imputed'])['start'].count())

    for exp in ['INH1A', 'CP1B', 'CP1A', 'INH2B']:
        if exp == 'CP1A':
            exp = 'CP1'
        df.loc[df['experiment_str'] == exp, 'mouse_id_str'] = df.loc[df['experiment_str'] == exp, 'mouse_id'].apply(
            lambda x: [k for k, v in dict_label2int[exp]['mouse_id'].items() if v == x][0])
        df.loc[df['experiment_str'] == exp, 'treatment_letter'] = df.loc[
            df['experiment_str'] == exp, 'treatment_detail'].apply(
            lambda x: [k for k, v in dict_label2int[exp]['treatment_detail'].items() if v == x][0])

    tmp = df[df['experiment_str'].isin(['INH1A', 'INH2B'])].groupby(['index_sample', 'side', 'imputed'])['index'].agg('count').reset_index().pivot(values='index',
                                                                                            columns='imputed',
                                                                                            index=('index_sample', 'side'))
    print(f'On average over all the samples of INH1A and INH2B, we detected '
          f'{((tmp[tmp[False] >= 10].loc[:, True] - tmp.loc[:, False]) / tmp.loc[:, False] * 100).mean():.1f}% '
          f'more steps in the imputed data compared to non imputed')

    indices_to_discard_orig = tmp.reset_index().loc[tmp.reset_index()[False] <= threshold_count_steps, ['index_sample', 'side']]
    indices_to_discard_orig.loc[:, 'under_threshold'] = True
    indices_to_discard_orig.loc[:, 'imputed'] = False
    indices_to_discard_imputed = tmp.reset_index().loc[tmp.reset_index()[True] <= threshold_count_steps, ['index_sample', 'side']]
    indices_to_discard_imputed.loc[:, 'under_threshold'] = True
    indices_to_discard_imputed.loc[:, 'imputed'] = True

    df = pd.merge(df, pd.concat([indices_to_discard_orig, indices_to_discard_imputed]), on=['index_sample', 'side', 'imputed'], how='left')
    df.loc[df['under_threshold'] != True, 'under_threshold'] = False

    tmp = df[df['experiment_str'] == 'INH1A'].groupby(['index_sample', 'side', 'imputed'])['index'].agg('count').reset_index().pivot(values='index',
                                                                                            columns='imputed',
                                                                                            index=('index_sample', 'side'))
    print(f'On average over all the samples of INH1A, we detected '
          f'{((tmp[tmp[False] >= 10].loc[:, True] - tmp.loc[:, False]) / tmp.loc[:, False] * 100).mean():.1f}% '
          f'more steps in the imputed data compared to non imputed')


    manual_df = pd.read_csv(os.path.join(featuresdir, 'manual_step_count_bij.csv'))
    step_count = df[df['experiment_str'] == 'INH1A'].groupby(['mouse_id_str', 'treatment_letter', 'side', 'imputed', 'index_sample'])[
        'index'].agg('count').reset_index()
    step_count = pd.merge(step_count, manual_df, how='left', on=['mouse_id_str', 'treatment_letter'])
    step_count.loc[:, 'diff_auto_manual'] = step_count['n_steps_manual'] - step_count['index']
    mean_step_count = step_count.groupby(['mouse_id_str', 'treatment_letter', 'imputed', 'index_sample'])['index'].agg(
        'mean').reset_index()
    mean_step_count = pd.merge(mean_step_count, manual_df, how='left', on=['mouse_id_str', 'treatment_letter'])
    mean_step_count.loc[:, 'diff_auto_manual'] = mean_step_count['n_steps_manual'] - mean_step_count['index']


    step_count.loc[:, 'diff_auto_manual%'] = (step_count['n_steps_manual'] - step_count['index']) / step_count['n_steps_manual'] * 100
    mean_step_count.loc[:, 'diff_auto_manual%'] = (mean_step_count['n_steps_manual'] - mean_step_count['index']) / mean_step_count['n_steps_manual'] * 100

    fig, ax = plt.subplots(1, 1, figsize=(3, 7))
    hue_plot_params = dict(data=mean_step_count, x='imputed', y='diff_auto_manual%', ax=ax)
    pairs = [(False, True)]
    sns.boxplot(**hue_plot_params)
    annotator = Annotator(pairs=pairs, **hue_plot_params)
    annotator.configure(test="t-test_paired").apply_and_annotate()
    print(f'N samples: {len(mean_step_count)} for comparison auto vs manual')
    sns.stripplot(data=mean_step_count, x='imputed', y='diff_auto_manual%', jitter=.1, palette=['skyblue', 'gold'])
    plt.ylim(-20, 85)
    plt.suptitle(f'Paired Ttest p-val = {annotator.annotations[0].data.pvalue:.2E}')
    plt.savefig(os.path.join(featuresdir, f'diff_percent_count_steps_manual-auto_meanRightLeft_INH1A_FL2.png'))
    plt.savefig(os.path.join(featuresdir, f'diff_percent_count_steps_manual-auto_meanRightLeft_INH1A_FL2.svg'))

    plt.figure(figsize=(3, 7))
    sns.boxplot(data=step_count, x='imputed', y='diff_auto_manual%')
    sns.stripplot(data=step_count, x='imputed', y='diff_auto_manual%', jitter=.1, palette=['skyblue', 'gold'])
    plt.ylim(-20, 85)
    plt.savefig(os.path.join(featuresdir, f'diff_percent_count_steps_manual-auto_RightLeft_INH1A_FL2.png'))
    plt.savefig(os.path.join(featuresdir, f'diff_percent_count_steps_manual-auto_RightLeft_INH1A_FL2.svg'))


    count_steps = df.groupby(['experiment_str', 'treatment_str', 'index_sample', 'side', 'imputed'])['start'].agg('count').reset_index()
    percent_nan_ankle_df = percent_nan_per_kp_df.loc[percent_nan_per_kp_df['keypoint'].isin(['right_ankle', 'left_ankle'])]
    percent_nan_ankle_df.loc[percent_nan_ankle_df['keypoint'] == 'left_ankle', 'side'] = 'left'
    percent_nan_ankle_df.loc[percent_nan_ankle_df['keypoint'] == 'right_ankle', 'side'] = 'right'
    count_steps = pd.merge(count_steps, percent_nan_ankle_df, how='left', on=['index_sample', 'imputed', 'side'])
    count_steps.loc[:, 'minute_data'] = (100 - count_steps['percent_nan']) / 100
    count_steps.loc[:, 'step_per_minute'] = count_steps['start'] / count_steps['minute_data']

    sns.set_style('ticks')
    bins_step_count = np.array([0, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111], dtype=float)
    table = np.vstack([np.histogram(count_steps.loc[count_steps['imputed'] == False, 'start'].values, bins=bins_step_count)[0],
                       np.histogram(count_steps.loc[count_steps['imputed'] == True, 'start'].values, bins=bins_step_count)[0]]).T
    res = chi2_contingency(table, correction=False)
    plt.figure()
    sns.histplot(data=count_steps, x='start', hue='imputed', multiple="dodge", shrink=0.8, kde=True, bins=bins_step_count)
    plt.suptitle(f'Chi2 contengency test p-val = {res.pvalue:.2E}')
    plt.xlabel('Step count per 1-min recording')
    plt.savefig(os.path.join(featuresdir, f'hist_count_steps_per_recording_FL2.png'))
    plt.savefig(os.path.join(featuresdir, f'hist_count_steps_per_recording_FL2.svg'))

    for exp in [['INH1A'], ['INH2B'], ['INH1A', 'INH2B']]:
        plt.figure()
        max_ = count_steps.loc[(count_steps['experiment_str'].isin(exp)) * (count_steps['imputed'] == True), 'start'].max()
        argmax_ = np.argmax(bins_step_count > max_)
        table = np.vstack(
            [np.histogram(count_steps.loc[(count_steps['experiment_str'].isin(exp)) * (count_steps['imputed'] == False), 'start'].values,
                          bins=bins_step_count[:argmax_ + 1])[0],
             np.histogram(count_steps.loc[(count_steps['experiment_str'].isin(exp)) * (count_steps['imputed'] == True), 'start'].values,
                          bins=bins_step_count[:argmax_ + 1])[0]]).T
        res = chi2_contingency(table, correction=False)
        sns.histplot(data=count_steps.loc[count_steps['experiment_str'].isin(exp)],
                     x='start', hue='imputed', multiple="dodge", shrink=0.8, kde=True, bins=bins_step_count)
        plt.xlabel('Step count per 1-min recording')
        plt.suptitle("-".join(exp) + f'\nChi2 contengency test p-val = {res.pvalue:.2E}')
        plt.savefig(os.path.join(featuresdir, f'hist_count_steps_per_recording_FL2_{"-".join(exp)}.png'))
        plt.savefig(os.path.join(featuresdir, f'hist_count_steps_per_recording_FL2_{"-".join(exp)}.svg'))


    for line in df.loc[df['imputed']].iterrows():
        exp, treatment, side = line[1]['experiment_str'], line[1]['treatment_str'], line[1]['side']
        start, stop = line[1]['start'], line[1]['stop']
        starts, stops = df.loc[
            (df['experiment_str'] == exp) * (df['treatment_str'] == treatment) * (df['side'] == side) * (
                        df['imputed'] == False), ['start', 'stop']].values.T
        potential_id = np.unique([np.argmin(np.abs(starts - start)), np.argmin(np.abs(stops - stop))])
        duplicate = False
        for pid in potential_id:
            overlap = (min(stops[pid], stop) - max(starts[pid], start)) / (
                        max(stops[pid], stop) - min(starts[pid], start))
            if overlap > 0.8:
                duplicate = True
        if duplicate:
            df.loc[line[0], 'imputed_only'] = False
        else:
            df.loc[line[0], 'imputed_only'] = True

    print(np.unique(df['imputed'], return_counts=True), np.unique(df['imputed_only'], return_counts=True))

    df_notimputed = df.loc[df['imputed'] == False]
    df_notimputed.loc[:, 'before_or_after'] = 'before'
    df.loc[:, 'before_or_after'] = 'after'
    df = pd.concat([df, df_notimputed])

    suffix = '_test_bw_treatments_all'
    drugs = [
        'vehicle_INH', # ctrl
        'PF3845 10mg/kg', # drug A
        'MJN110 1.25mg/kg' # drug B
        ]
    exp = 'INH1A'
    mask = (df['experiment_str'] == exp) * (df['under_threshold'] == False)
    fig, ax = plt.subplots(5, 2, figsize=(3 * 2, 4 * 5))
    for iside, side in enumerate(['left', 'right']):
        print(f"{side} N SAMPLES: {df[mask * (df['side'] == side)].groupby(['treatment_str', 'before_or_after'])['mouse_id'].count()}", end='\n\n')
        for imeasure, measure in enumerate([
            'swing_height', 'swing_direct_distance',
           'swing_total_distance',
            'swing_mean_speed',
           'swing_duration']):
            print(f"--- {measure} on {side} side")
            hue_plot_params = dict(data=df[mask * (df['side'] == side)], y=measure, x='treatment_str', hue='before_or_after', order=drugs, ax=ax[imeasure, iside], hue_order=['before', 'after'])
            pairs = [((drugs[0], 'before'), (t, 'before')) for t in drugs[1:]] + [((drugs[0], 'after'), (t, 'after')) for t in drugs[1:]]
            sns.boxplot(**hue_plot_params)
            plt.suptitle(f'{exp}')
            annotator = Annotator(pairs=pairs, **hue_plot_params)
            annotator.configure(test="t-test_ind").apply_and_annotate()
            ax[imeasure, iside].legend([], [], frameon=False)
    plt.tight_layout()

    plt.savefig(os.path.join(featuresdir, f'compare_boxplot_{exp}_all_treatments{suffix}_FL2.png'))
    plt.savefig(os.path.join(featuresdir, f'compare_boxplot_{exp}_all_treatments{suffix}_FL2.svg'))


"""
FULL OUTPUT

treatment_str     side   imputed
AM251 + MJN110    left   False       90
                         True       137
                  right  False       66
                         True       110
AM251 + PF3845    left   False       24
                         True        38
                  right  False       30
                         True        39
AM251 3mg/kg      left   False       69
                         True       128
                  right  False       78
                         True       132
CP55,940 0.01     left   False      102
                         True       131
                  right  False       71
                         True       106
CP55,940 0.03     left   False       95
                         True       149
                  right  False      163
                         True       183
CP55,940 0.3      left   False       75
                         True        97
                  right  False       36
                         True        74
MJN110 1.25mg/kg  left   False      654
                         True       826
                  right  False      487
                         True       767
PF3845 10mg/kg    left   False      228
                         True       352
                  right  False      239
                         True       348
vehicle_CP        left   False      158
                         True       201
                  right  False      185
                         True       224
vehicle_INH       left   False      516
                         True       708
                  right  False      447
                         True       674
Name: start, dtype: int64
On average over all the samples of INH1A and INH2B, we detected 57.2% more steps in the imputed data compared to non imputed
On average over all the samples of INH1A, we detected 34.3% more steps in the imputed data compared to non imputed
p-value annotation legend:
      ns: p <= 1.00e+00
       *: 1.00e-02 < p <= 5.00e-02
      **: 1.00e-03 < p <= 1.00e-02
     ***: 1.00e-04 < p <= 1.00e-03
    ****: p <= 1.00e-04

False vs. True: t-test paired samples, P_val:7.706e-05 t=4.894e+00
N samples: 44 for comparison auto vs manual
/home/france/Documents/DISK_paper_code/fig6_step_detection/compare_before_after_imputation_step_features.py:152: FutureWarning: Passing `palette` without assigning `hue` is deprecated.
  sns.stripplot(data=mean_step_count, x='imputed', y='diff_auto_manual%', jitter=.1, palette=['skyblue', 'gold'])
/home/france/Documents/DISK_paper_code/fig6_step_detection/compare_before_after_imputation_step_features.py:160: FutureWarning: Passing `palette` without assigning `hue` is deprecated.
  sns.stripplot(data=step_count, x='imputed', y='diff_auto_manual%', jitter=.1, palette=['skyblue', 'gold'])
/home/france/Documents/DISK_paper_code/fig6_step_detection/compare_before_after_imputation_step_features.py:168: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  percent_nan_ankle_df.loc[percent_nan_ankle_df['keypoint'] == 'left_ankle', 'side'] = 'left'
(array([False,  True]), array([3813, 5424])) (array([nan, False, True, ..., nan, nan, True], dtype=object), array([   1, 3965, 1120, ...,    1,    1,    1]))
/home/france/Documents/DISK_paper_code/fig6_step_detection/compare_before_after_imputation_step_features.py:225: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  df_notimputed.loc[:, 'before_or_after'] = 'before'
left N SAMPLES: treatment_str     before_or_after
MJN110 1.25mg/kg  after              1151
                  before              552
PF3845 10mg/kg    after               378
                  before              157
vehicle_INH       after               676
                  before              313
Name: mouse_id, dtype: int64

--- swing_height on left side
p-value annotation legend:
      ns: p <= 1.00e+00
       *: 1.00e-02 < p <= 5.00e-02
      **: 1.00e-03 < p <= 1.00e-02
     ***: 1.00e-04 < p <= 1.00e-03
    ****: p <= 1.00e-04

vehicle_INH_before vs. PF3845 10mg/kg_before: t-test independent samples, P_val:3.715e-01 t=8.945e-01
vehicle_INH_after vs. PF3845 10mg/kg_after: t-test independent samples, P_val:1.807e-01 t=1.339e+00
vehicle_INH_before vs. MJN110 1.25mg/kg_before: t-test independent samples, P_val:6.351e-01 t=4.747e-01
vehicle_INH_after vs. MJN110 1.25mg/kg_after: t-test independent samples, P_val:3.941e-01 t=8.524e-01
--- swing_direct_distance on left side
p-value annotation legend:
      ns: p <= 1.00e+00
       *: 1.00e-02 < p <= 5.00e-02
      **: 1.00e-03 < p <= 1.00e-02
     ***: 1.00e-04 < p <= 1.00e-03
    ****: p <= 1.00e-04

vehicle_INH_before vs. PF3845 10mg/kg_before: t-test independent samples, P_val:3.775e-02 t=2.083e+00
vehicle_INH_after vs. PF3845 10mg/kg_after: t-test independent samples, P_val:1.048e-04 t=3.894e+00
vehicle_INH_before vs. MJN110 1.25mg/kg_before: t-test independent samples, P_val:4.866e-07 t=-5.070e+00
vehicle_INH_after vs. MJN110 1.25mg/kg_after: t-test independent samples, P_val:1.608e-12 t=-7.114e+00
--- swing_total_distance on left side
p-value annotation legend:
      ns: p <= 1.00e+00
       *: 1.00e-02 < p <= 5.00e-02
      **: 1.00e-03 < p <= 1.00e-02
     ***: 1.00e-04 < p <= 1.00e-03
    ****: p <= 1.00e-04

vehicle_INH_before vs. PF3845 10mg/kg_before: t-test independent samples, P_val:5.658e-02 t=1.911e+00
vehicle_INH_after vs. PF3845 10mg/kg_after: t-test independent samples, P_val:2.705e-04 t=3.654e+00
vehicle_INH_before vs. MJN110 1.25mg/kg_before: t-test independent samples, P_val:1.598e-05 t=-4.339e+00
vehicle_INH_after vs. MJN110 1.25mg/kg_after: t-test independent samples, P_val:2.870e-09 t=-5.968e+00
--- swing_mean_speed on left side
p-value annotation legend:
      ns: p <= 1.00e+00
       *: 1.00e-02 < p <= 5.00e-02
      **: 1.00e-03 < p <= 1.00e-02
     ***: 1.00e-04 < p <= 1.00e-03
    ****: p <= 1.00e-04

vehicle_INH_before vs. PF3845 10mg/kg_before: t-test independent samples, P_val:1.494e-03 t=3.195e+00
vehicle_INH_after vs. PF3845 10mg/kg_after: t-test independent samples, P_val:4.480e-08 t=5.511e+00
vehicle_INH_before vs. MJN110 1.25mg/kg_before: t-test independent samples, P_val:1.233e-08 t=-5.751e+00
vehicle_INH_after vs. MJN110 1.25mg/kg_after: t-test independent samples, P_val:1.084e-15 t=-8.089e+00
--- swing_duration on left side
p-value annotation legend:
      ns: p <= 1.00e+00
       *: 1.00e-02 < p <= 5.00e-02
      **: 1.00e-03 < p <= 1.00e-02
     ***: 1.00e-04 < p <= 1.00e-03
    ****: p <= 1.00e-04

vehicle_INH_before vs. PF3845 10mg/kg_before: t-test independent samples, P_val:3.952e-04 t=-3.569e+00
vehicle_INH_after vs. PF3845 10mg/kg_after: t-test independent samples, P_val:1.309e-06 t=-4.867e+00
vehicle_INH_before vs. MJN110 1.25mg/kg_before: t-test independent samples, P_val:1.960e-02 t=2.338e+00
vehicle_INH_after vs. MJN110 1.25mg/kg_after: t-test independent samples, P_val:4.672e-04 t=3.505e+00
right N SAMPLES: treatment_str     before_or_after
MJN110 1.25mg/kg  after              920
                  before             377
PF3845 10mg/kg    after              372
                  before             160
vehicle_INH       after              639
                  before             289
Name: mouse_id, dtype: int64

--- swing_height on right side
p-value annotation legend:
      ns: p <= 1.00e+00
       *: 1.00e-02 < p <= 5.00e-02
      **: 1.00e-03 < p <= 1.00e-02
     ***: 1.00e-04 < p <= 1.00e-03
    ****: p <= 1.00e-04

vehicle_INH_before vs. PF3845 10mg/kg_before: t-test independent samples, P_val:2.710e-02 t=2.217e+00
vehicle_INH_after vs. PF3845 10mg/kg_after: t-test independent samples, P_val:1.254e-03 t=3.235e+00
vehicle_INH_before vs. MJN110 1.25mg/kg_before: t-test independent samples, P_val:9.778e-01 t=2.782e-02
vehicle_INH_after vs. MJN110 1.25mg/kg_after: t-test independent samples, P_val:5.093e-02 t=1.954e+00
--- swing_direct_distance on right side
p-value annotation legend:
      ns: p <= 1.00e+00
       *: 1.00e-02 < p <= 5.00e-02
      **: 1.00e-03 < p <= 1.00e-02
     ***: 1.00e-04 < p <= 1.00e-03
    ****: p <= 1.00e-04

vehicle_INH_before vs. PF3845 10mg/kg_before: t-test independent samples, P_val:3.926e-02 t=2.067e+00
vehicle_INH_after vs. PF3845 10mg/kg_after: t-test independent samples, P_val:2.459e-04 t=3.680e+00
vehicle_INH_before vs. MJN110 1.25mg/kg_before: t-test independent samples, P_val:6.635e-08 t=-5.463e+00
vehicle_INH_after vs. MJN110 1.25mg/kg_after: t-test independent samples, P_val:1.214e-14 t=-7.790e+00
--- swing_total_distance on right side
p-value annotation legend:
      ns: p <= 1.00e+00
       *: 1.00e-02 < p <= 5.00e-02
      **: 1.00e-03 < p <= 1.00e-02
     ***: 1.00e-04 < p <= 1.00e-03
    ****: p <= 1.00e-04

vehicle_INH_before vs. PF3845 10mg/kg_before: t-test independent samples, P_val:2.424e-03 t=3.050e+00
vehicle_INH_after vs. PF3845 10mg/kg_after: t-test independent samples, P_val:1.378e-06 t=4.857e+00
vehicle_INH_before vs. MJN110 1.25mg/kg_before: t-test independent samples, P_val:5.953e-05 t=-4.041e+00
vehicle_INH_after vs. MJN110 1.25mg/kg_after: t-test independent samples, P_val:2.026e-08 t=-5.639e+00
--- swing_mean_speed on right side
p-value annotation legend:
      ns: p <= 1.00e+00
       *: 1.00e-02 < p <= 5.00e-02
      **: 1.00e-03 < p <= 1.00e-02
     ***: 1.00e-04 < p <= 1.00e-03
    ****: p <= 1.00e-04

vehicle_INH_before vs. PF3845 10mg/kg_before: t-test independent samples, P_val:2.878e-04 t=3.655e+00
vehicle_INH_after vs. PF3845 10mg/kg_after: t-test independent samples, P_val:9.391e-09 t=5.790e+00
vehicle_INH_before vs. MJN110 1.25mg/kg_before: t-test independent samples, P_val:9.313e-09 t=-5.817e+00
vehicle_INH_after vs. MJN110 1.25mg/kg_after: t-test independent samples, P_val:9.586e-16 t=-8.117e+00
--- swing_duration on right side
p-value annotation legend:
      ns: p <= 1.00e+00
       *: 1.00e-02 < p <= 5.00e-02
      **: 1.00e-03 < p <= 1.00e-02
     ***: 1.00e-04 < p <= 1.00e-03
    ****: p <= 1.00e-04

vehicle_INH_before vs. PF3845 10mg/kg_before: t-test independent samples, P_val:1.603e-01 t=-1.407e+00
vehicle_INH_after vs. PF3845 10mg/kg_after: t-test independent samples, P_val:7.358e-02 t=-1.791e+00
vehicle_INH_before vs. MJN110 1.25mg/kg_before: t-test independent samples, P_val:4.563e-04 t=3.523e+00
vehicle_INH_after vs. MJN110 1.25mg/kg_after: t-test independent samples, P_val:5.667e-07 t=5.023e+00
"""