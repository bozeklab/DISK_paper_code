import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr
import numpy as np
import importlib.util
from glob import glob
from utils import plot_save, read_constant_file





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