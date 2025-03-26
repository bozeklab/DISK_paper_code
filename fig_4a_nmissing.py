import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

###########################################################################
### N MISSING PLOT

df = pd.read_csv('/home/france/Dropbox/Dropbox/2021_Koeln/behavior/fl2_n_missing_results.csv')
sns.lineplot(df, x='n_missing', hue='method', y='test error',
             hue_order=['linear_interp', 'GRU', 'transformer_baseline'], palette=['gray', 'gold', 'orangered'])
plt.savefig(f'/home/france/Dropbox/Dropbox/2021_Koeln/bogna/fl2_RMSE_n_missing_plot.svg')