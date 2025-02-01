
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

from behaviourrepresentation.utils.experiment import BognaExp

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
    file = 'FL2'
    basedir = f'/home/france/Mounted_dir/results_behavior/datasets/INHCP_{file}_keypoints_60stride30/'
    featuresdir = os.path.join(basedir, 'bogna_features/')
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

    plt.figure()
    sns.boxplot(data=percent_nan_per_kp_df, hue='imputed', y='percent_nan', x='keypoint')
    plt.savefig(os.path.join(featuresdir, f'percent_nan_train_before_after_imputation_per_keypoint_{file}.png'))
    plt.savefig(os.path.join(featuresdir, f'percent_nan_train_before_after_imputation_per_keypoint_{file}.svg'))

    percent_nan_imputed = [np.sum(np.isnan(x[x != -1])) / len(x[x != -1]) * 100 for x in X_imputed]
    percent_nan_before = [np.sum(np.isnan(x[x != -1])) / len(x[x != -1]) * 100 for x in X_before]
    percent_nan_df = pd.DataFrame(columns=['index_sample', 'imputed', 'percent_nan'], data=np.vstack(
        [np.concatenate([np.zeros(len(percent_nan_before), dtype=bool),
                         np.ones(len(percent_nan_imputed), dtype=bool)]),
         np.concatenate([percent_nan_before,
                         percent_nan_imputed]),
         np.concatenate([np.arange(len(percent_nan_before)),
                         np.arange(len(percent_nan_imputed))])
         ]).T)
    percent_nan_df = percent_nan_df.reset_index().rename({'index': 'index_sample'}, axis=1)
    plt.figure()
    sns.boxplot(data=percent_nan_df, x='imputed', y='percent_nan')
    plt.savefig(os.path.join(featuresdir, f'percent_nan_train_before_after_imputation_{file}.png'))
    plt.savefig(os.path.join(featuresdir, f'percent_nan_train_before_after_imputation_{file}.svg'))

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
            lambda x: [k for k, v in BognaExp.dict_label2int[exp]['mouse_id'].items() if v == x][0])
        df.loc[df['experiment_str'] == exp, 'treatment_letter'] = df.loc[
            df['experiment_str'] == exp, 'treatment'].apply(
            lambda x: [k for k, v in BognaExp.dict_label2int[exp]['treatment'].items() if v == x][0])

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


    if 'FL2' in featuresdir:
        manual_df = pd.read_csv(os.path.join(featuresdir, 'manual_step_count_bij.csv'))
        step_count = df[df['experiment_str'] == 'INH1A'].groupby(['mouse_id_str', 'treatment_letter', 'side', 'imputed', 'index_sample'])[
            'index'].agg('count').reset_index()
        step_count = pd.merge(step_count, manual_df, how='left', on=['mouse_id_str', 'treatment_letter'])
        step_count.loc[:, 'diff_auto_manual'] = step_count['n_steps_manual'] - step_count['index']
        mean_step_count = step_count.groupby(['mouse_id_str', 'treatment_letter', 'imputed', 'index_sample'])['index'].agg(
            'mean').reset_index()
        mean_step_count = pd.merge(mean_step_count, manual_df, how='left', on=['mouse_id_str', 'treatment_letter'])
        mean_step_count.loc[:, 'diff_auto_manual'] = mean_step_count['n_steps_manual'] - mean_step_count['index']

        # plt.figure(figsize=(3, 6))
        # sns.boxplot(data=mean_step_count, x='imputed', y='diff_auto_manual')
        # sns.stripplot(data=mean_step_count, x='imputed', y='diff_auto_manual', jitter=.1, palette=['skyblue', 'gold'])
        # plt.ylim(-10, 105)
        # plt.savefig(os.path.join(featuresdir, f'diff_count_steps_manual-auto_meanRightLeft_INH1A_{file}.png'))
        # plt.savefig(os.path.join(featuresdir, f'diff_count_steps_manual-auto_meanRightLeft_INH1A_{file}.svg'))

        step_count.loc[:, 'diff_auto_manual%'] = (step_count['n_steps_manual'] - step_count['index']) / step_count['n_steps_manual'] * 100
        mean_step_count.loc[:, 'diff_auto_manual%'] = (mean_step_count['n_steps_manual'] - mean_step_count['index']) / mean_step_count['n_steps_manual'] * 100

        fig, ax = plt.subplots(1, 1, figsize=(3, 7))
        hue_plot_params = dict(data=mean_step_count, x='imputed', y='diff_auto_manual%', ax=ax)
        pairs = [(False, True)]
        sns.boxplot(**hue_plot_params)
        annotator = Annotator(pairs=pairs, **hue_plot_params)
        annotator.configure(test="t-test_paired").apply_and_annotate()
        sns.stripplot(data=mean_step_count, x='imputed', y='diff_auto_manual%', jitter=.1, palette=['skyblue', 'gold'])
        plt.ylim(-20, 85)
        plt.suptitle(f'Paired Ttest p-val = {annotator.annotations[0].data.pvalue:.2E}')
        plt.savefig(os.path.join(featuresdir, f'diff_percent_count_steps_manual-auto_meanRightLeft_INH1A_{file}.png'))
        plt.savefig(os.path.join(featuresdir, f'diff_percent_count_steps_manual-auto_meanRightLeft_INH1A_{file}.svg'))

        plt.figure(figsize=(3, 7))
        sns.boxplot(data=step_count, x='imputed', y='diff_auto_manual%')
        sns.stripplot(data=step_count, x='imputed', y='diff_auto_manual%', jitter=.1, palette=['skyblue', 'gold'])
        plt.ylim(-20, 85)
        plt.savefig(os.path.join(featuresdir, f'diff_percent_count_steps_manual-auto_RightLeft_INH1A_{file}.png'))
        plt.savefig(os.path.join(featuresdir, f'diff_percent_count_steps_manual-auto_RightLeft_INH1A_{file}.svg'))

    count_steps = df.groupby(['treatment_str', 'index_sample', 'side', 'imputed'])['start'].agg('count').reset_index()
    plt.figure()
    sns.boxplot(data=count_steps, y='start', hue='imputed', x='side')
    plt.savefig(os.path.join(featuresdir, f'diff_count_steps_auto_RightLeft_allTreatments_{file}.png'))
    plt.savefig(os.path.join(featuresdir, f'diff_count_steps_auto_RightLeft_allTreatments_{file}.svg'))

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
    plt.savefig(os.path.join(featuresdir, f'hist_count_steps_per_recording_{file}.png'))
    plt.savefig(os.path.join(featuresdir, f'hist_count_steps_per_recording_{file}.svg'))

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
        plt.savefig(os.path.join(featuresdir, f'hist_count_steps_per_recording_{file}_{"-".join(exp)}.png'))
        plt.savefig(os.path.join(featuresdir, f'hist_count_steps_per_recording_{file}_{"-".join(exp)}.svg'))

    for exp in [['INH1A'], ['INH2B'], ['INH1A', 'INH2B']]:
        mask = (df['experiment_str'].isin(exp)) * (df['treatment_str'] == 'vehicle_INH')
        exp_name = exp[0] if len(exp) == 1 else '+'.join(exp)

        count_steps = df[mask].groupby(['treatment_str', 'index_sample', 'side', 'imputed', '10second_window'])['start'].agg('count').reset_index()
        cum_count_steps = []
        for i in range(6):
            tmp = count_steps[count_steps['10second_window'] <= i].groupby(['treatment_str', 'index_sample', 'side', 'imputed'])['start'].agg('sum').reset_index()
            tmp.loc[:, 'Time (sec)'] = i * 10
            cum_count_steps.append(tmp)
        cum_count_steps = pd.concat(cum_count_steps)

        plt.figure()
        sns.lineplot(data=cum_count_steps, x='Time (sec)', y='start', hue='imputed', errorbar=('se', 1), err_style='bars',)
        plt.axhline(y=20, c='grey')
        plt.suptitle(exp_name)
        plt.ylabel('Cumulative step count')
        plt.savefig(os.path.join(featuresdir, f'cumulative_step_count_{exp_name}_{file}.png'))
        plt.savefig(os.path.join(featuresdir, f'cumulative_step_count_{exp_name}_{file}.svg'))
    # th_min_nb_steps = 1
    # sample_count_table = (df[mask].groupby(['index_sample', 'treatment_str', 'side', 'imputed', '10second_window'])['start'].agg('count') >= th_min_nb_steps).reset_index().rename({'start': 'filter'}, axis=1)
    # df_plus = pd.merge(df, sample_count_table, on=['index_sample', 'imputed', 'treatment_str', 'side', '10second_window'], how='right')
    # mask = (df_plus['experiment_str'] == exp) * (df_plus['filter'])

    # count_steps = df[mask].groupby(['treatment_str', 'index_sample', 'side', 'imputed'])['start'].agg('count').reset_index()
    # plt.figure()
    # plt.suptitle(exp)
    # sns.boxplot(data=count_steps, y='start', hue='imputed', x='side')
    # plt.savefig(os.path.join(featuresdir, f'diff_count_steps_auto_RightLeft_{exp}_{file}.png'))
    # plt.savefig(os.path.join(featuresdir, f'diff_count_steps_auto_RightLeft_{exp}_{file}.svg'))
    #
    # count_steps = df[mask].groupby(['treatment_str', 'index_sample', 'side', 'imputed'])['start'].agg('count').reset_index()
    # plt.figure()
    # plt.suptitle(exp)
    # sns.histplot(data=count_steps, x='start', hue='imputed', multiple="dodge", shrink=0.8)
    # plt.xlabel('Step count per 1-min recording')
    # plt.savefig(os.path.join(featuresdir, f'hist_count_steps_per_recording_{exp}_{file}.png'))
    # plt.savefig(os.path.join(featuresdir, f'hist_count_steps_per_recording_{exp}_{file}.svg'))

    for line in df.loc[mask * df['imputed']].iterrows():
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

    drugs = [
        'vehicle_INH',
        'PF3845 10mg/kg',
        'MJN110 1.25mg/kg']
    exp = 'INH1A'
    mask = (df['experiment_str'] == exp) * (df['under_threshold'] == False)
    fig, ax = plt.subplots(5, 2, figsize=(3 * 2, 4 * 5))
    for iside, side in enumerate(['left', 'right']):
        for imeasure, measure in enumerate([
            'swing_height', 'swing_direct_distance',
           'swing_total_distance',
            'swing_mean_speed',
           'swing_duration']):
            print(f'--- {measure} on {side} side')
            hue_plot_params = dict(data=df[mask * (df['side'] == side)], hue='imputed', y=measure, x='treatment_str', order=drugs, ax=ax[imeasure, iside])
            pairs = [((drugs[0], False), (t, False)) for t in drugs[1:]] + [((drugs[0], True), (t, True)) for t in drugs[1:]]
            suffix = '_test_bw_treatments'
            sns.boxplot(**hue_plot_params)
            plt.suptitle(f'{exp} {side} {measure}')
            annotator = Annotator(pairs=pairs, **hue_plot_params)
            annotator.configure(test="t-test_ind").apply_and_annotate()
            ax[imeasure, iside].legend([], [], frameon=False)
    plt.tight_layout()

    plt.savefig(os.path.join(featuresdir, f'compare_boxplot_{exp}_all_treatments{suffix}_{file}.png'))
    plt.savefig(os.path.join(featuresdir, f'compare_boxplot_{exp}_all_treatments{suffix}_{file}.svg'))

    drugs = [
        'vehicle_INH',
        'PF3845 10mg/kg',
        'MJN110 1.25mg/kg']
    exp = 'INH1A'
    mask = (df['experiment_str'] == exp) * (df['under_threshold'] == False)
    fig, ax = plt.subplots(5, 2, figsize=(3 * 2, 4 * 5))
    for iside, side in enumerate(['left', 'right']):
        for imeasure, measure in enumerate(['swing_height', 'swing_direct_distance',
           'swing_total_distance', 'swing_mean_speed',
           'swing_duration']):
            hue_plot_params = dict(data=df[mask * (df['side'] == side) * (df['imputed_only'].isin([True, np.nan]))], hue='imputed', y=measure, x='treatment_str', order=drugs, ax=ax[imeasure, iside])
            pairs = [((t, False), (t, True)) for t in drugs]
            suffix = 'test_bw_imputed'
            sns.boxplot(**hue_plot_params)
            # sns.stripplot(**hue_plot_params, palette=['skyblue', 'gold'], dodge=True, jitter=.3, alpha=.8, marker="$\circ$")
            plt.suptitle(f'{exp} {side} {measure}')
            annotator = Annotator(pairs=pairs, **hue_plot_params)
            annotator.configure(test="t-test_ind").apply_and_annotate()
            ax[imeasure, iside].legend([], [], frameon=False)
    plt.tight_layout()

    plt.savefig(os.path.join(featuresdir, f'compare_boxplot_{exp}_all_treatments_{suffix}_{file}.png'))
    plt.savefig(os.path.join(featuresdir, f'compare_boxplot_{exp}_all_treatments_{suffix}_{file}.svg'))
    # plt.close()
