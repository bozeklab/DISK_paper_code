import os, sys
import tqdm
from glob import glob
import seaborn as sns
import numpy as np
import pandas as pd
from skimage.io import imread, imsave
import argparse
import matplotlib.pyplot as plt

import matplotlib

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


if __name__ == '__main__':

    ##################################################################################################
    ### ERROR BY ACTION / DIST

    ##########################################################################################################
    ### CHOOSE DATASET BY SUPPLYING THE COMMANDLINE ARGUMENT
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset', type=str,
                        help='dataset name', choices=['Mocap', 'MABe'])

    args = parser.parse_args()
    ##########################################################################################################

    if args.dataset == 'MABe':
        input_file = '/home/france/frose1_ada/results_behavior/models/MABe_10-44-22_transformer_NLL/MABE_task1_60stride60_metadata_20250317_origcoords.csv'
        columns = ['movement', 'movement_mouse1', 'movement_mouse2',
       'movement_mouse1-mouse2', 'speed_xy', 'dist_bw_mice', 'angle_2mice',
       'angle_base', 'angle_mouse1', 'angle_mouse2', 'periodicity_max']
    elif args.dataset == 'Mocap':
        input_file = '/home/france/frose1_ada/results_behavior/models/03-10-24_transformer_NLL/Mocap_keypoints_60_stride30_metadata_20250317_origcoords.csv'
        columns = ['movement', 'upside_down', 'speed_xy', 'speed_z',
       'average_height', 'back_length', 'dist_barycenter_shoulders',
       'height_shoulders', 'angleXY_shoulders', 'dist_bw_knees',
       'dist_knees_shoulders', 'angle_back_base', 'periodicity_max']
    else:
        raise NotImplementedError

    output_dir = '/home/france/Dropbox/Dropbox/2021_Koeln/bogna/DISK_performance_vs_action_202503'
    metric = 'mpjpe'
    ## MABe
    df = pd.read_csv(input_file)
    df.loc[:, 'n_missing_bin'] = df['n_missing'].apply(binning)

    if 'swapped' in df.columns:
        mask = df['swapped'] == False
    else:
        mask = np.ones(df.shape[0], dtype=bool)

    plt.figure()
    plt.hist(df.loc[mask, metric], bins=100)
    plt.ylabel(metric)
    plt.savefig(os.path.join(output_dir, f'hist_{args.dataset}_{metric}.svg'))
    plt.savefig(os.path.join(output_dir, f'hist_{args.dataset}_{metric}.png'))

    plt.figure('histplot_action_vs_{metric}')
    sns.histplot(
        df[mask], x=metric, hue="action_str",
        stat="density", common_norm=False, bins=30,
    )
    plt.savefig(os.path.join(output_dir, f'hist_{args.dataset}_{metric}_vs_action.svg'))
    plt.savefig(os.path.join(output_dir, f'hist_{args.dataset}_{metric}_vs_action.png'))

    plt.figure(f'action_vs_{metric}')
    sns.barplot(data=df[mask], x='action_str', y=metric)
    plt.savefig(os.path.join(output_dir, f'barplot_{args.dataset}_{metric}_vs_action.svg'))
    plt.savefig(os.path.join(output_dir, f'barplot_{args.dataset}_{metric}_vs_action.png'))

    plt.figure('action_vs_{metric}_high_n_missing_sup40')
    sns.barplot(data=df[(mask) * (df['n_missing_bin'] > 4)], x='action_str', y=metric)
    plt.savefig(os.path.join(output_dir, f'barplot_{args.dataset}_{metric}_vs_action_high_n_missing_sup40.svg'))
    plt.savefig(os.path.join(output_dir, f'barplot_{args.dataset}_{metric}_vs_action_high_n_missing_sup40.png'))

    plt.figure('action_vs_n_missing')
    sns.lineplot(data=df[df['swapped']==False], x='n_missing_bin', hue='action_str', y=metric)
    plt.savefig(os.path.join(output_dir, f'lineplot_{args.dataset}_{metric}_vs_action_vs_n_missing.svg'))
    plt.savefig(os.path.join(output_dir, f'lineplot_{args.dataset}_{metric}_vs_action_vs_n_missing.png'))

    if 'dist_bw_mice' in df.columns:
        df.loc[:, 'dist_bw_mice_bin'] = df['dist_bw_mice'].apply(binning_dist)
        plt.figure('action_vs_dist_bw_animals')
        sns.lineplot(data=df[df['swapped']==False], x='dist_bw_mice_bin', hue='action_str', y=metric)
        plt.savefig(os.path.join(output_dir, f'lineplot_{args.dataset}_{metric}_vs_action_vs_dist_bw_animals.svg'))
        plt.savefig(os.path.join(output_dir, f'lineplot_{args.dataset}_{metric}_vs_action_vs_dist_bw_animals.png'))

        plt.figure('action_vs_n_missing_high_dist')
        sns.lineplot(data=df[(df['swapped']==False) * (df['dist_bw_mice_bin'] > 3)], x='n_missing_bin', hue='action_str', y=metric)
        plt.savefig(os.path.join(output_dir, f'lineplot_{args.dataset}_{metric}_vs_action_vs_n_missing_high_dist.svg'))
        plt.savefig(os.path.join(output_dir, f'lineplot_{args.dataset}_{metric}_vs_action_vs_n_missing_high_dist.png'))

        plt.figure('action_vs_n_missing_low_dist')
        sns.lineplot(data=df[(df['swapped']==False) * (df['dist_bw_mice_bin'] < 1)], x='n_missing_bin', hue='action_str', y=metric)
        plt.savefig(os.path.join(output_dir, f'lineplot_{args.dataset}_{metric}_vs_action_vs_n_missing_low_dist.svg'))
        plt.savefig(os.path.join(output_dir, f'lineplot_{args.dataset}_{metric}_vs_action_vs_n_missing_low_dist.png'))

    def bin_(values, n_min=10):
        bin_edges = np.sort(values)[n_min::n_min]
        bin_edges = np.insert(bin_edges, 0, np.min(values))
        bin_edges = np.insert(bin_edges, len(bin_edges), np.max(values))
        return np.array([np.argmax(v <= bin_edges) for v in values])

    plt.close('all')
    n_min = 150
    for c in columns:
        if c != 'n_missing':
            df.loc[:, f'bin_{c}'] = bin_(df[c].values, n_min=n_min)
            plt.figure(f'action_vs_{c}')
            sns.lineplot(data=df[(mask)], x=f'bin_{c}', estimator=np.mean,
                         hue='action_str', y=metric)
            plt.savefig(os.path.join(output_dir, f'lineplot_{args.dataset}_{metric}_vs_action_vs_{c}.svg'))
            plt.savefig(os.path.join(output_dir, f'lineplot_{args.dataset}_{metric}_vs_action_vs_{c}.png'))

            plt.figure(f'{c}')
            sns.lineplot(data=df[(mask)], x=f'bin_{c}', estimator=np.mean, y=metric)
            plt.savefig(os.path.join(output_dir, f'lineplot_{args.dataset}_{metric}_vs_{c}.svg'))
            plt.savefig(os.path.join(output_dir, f'lineplot_{args.dataset}_{metric}_vs_{c}.png'))
        else:
            plt.figure(f'action_vs_{c}')
            sns.lineplot(data=df[(mask)], x=c, estimator=np.mean,
                         hue='action_str', y=metric)
            plt.savefig(os.path.join(output_dir, f'lineplot_{args.dataset}_{metric}_vs_action_vs_{c}.svg'))
            plt.savefig(os.path.join(output_dir, f'lineplot_{args.dataset}_{metric}_vs_action_vs_{c}.png'))

            plt.figure(f'{c}')
            sns.lineplot(data=df[(mask)], x=c, estimator=np.mean, y=metric)
            plt.savefig(os.path.join(output_dir, f'lineplot_{args.dataset}_{metric}_vs_{c}.svg'))
            plt.savefig(os.path.join(output_dir, f'lineplot_{args.dataset}_{metric}_vs_{c}.png'))
    plt.close('all')
    print('stop')
