###########################################################
# Code to create plot graphs
# Author: Luca Adorni
# Date: January 2023
###########################################################

# 0. Setup -------------------------------------------------

import numpy as np
import pandas as pd
import os
import re
import pickle
import sys
from collections import Counter
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import matplotlib.dates as mdates
import random

pd.options.display.max_columns = 200
pd.options.display.max_rows = 1000
pd.set_option('max_info_columns', 200)
pd.set_option('expand_frame_repr', False)
pd.set_option('expand_frame_repr', True)
pd.set_option('max_colwidth',1000)
pd.set_option('display.width',None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Setup Repository
try:
    # Setup Repository
    with open("repo_info.txt", "r") as repo_info:
        path_to_repo = repo_info.readline()
except:
    path_to_repo = f"{os.getcwd()}/polpo/"
    sys.path.append(f"{os.getcwd()}/.local/bin") # import the temporary path where the server installed the module

  
print(path_to_repo)

path_to_data = f"{path_to_repo}data/"
path_to_raw = f"{path_to_data}raw/"
path_to_links = f"{path_to_data}links/"
path_to_processed = f"{path_to_data}processed/"
path_to_figures = f"{path_to_repo}figures/"
path_to_tables = f"{path_to_repo}tables/"

# define a function that saves figures
def save_fig(fig_id, tight_layout=True):
    # The path of the figures folder ./Figures/fig_id.png (fig_id is a variable that you specify 
    # when you call the function)
    path = os.path.join(fig_id + ".pdf") 
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='pdf', dpi=300)

# 1. Load Dataset ----------------------------------------------------


# Load our dataset
df = pd.read_pickle(f"{path_to_processed}final_df.pkl.gz", compression = 'gzip')

# 1. HISTOGRAM
# define plot dataframe
df_plot = df.groupby(['week_start','scree_name']).pred.mean()
df_plot = pd.DataFrame(df_plot)
# histogram
fig, ax = plt.subplots(figsize=(15, 10))
sns.histplot(data= df_plot, x = "pred", kde = False, stat = 'proportion', bins = 50, ax = ax)
sns.despine()
plt.ylabel('Week/User Share', fontsize = 35)
plt.xlabel('Predicted Polarization', fontsize = 35)
plt.yticks(fontsize = 30)
plt.xticks(fontsize = 30)
plt.axvline(0, linewidth = 1, alpha = 0.9, color='r', linestyle = '--')
save_fig(f'{path_to_figures}final/fig_2a_hist')

# 2. EVENT GRAPH
df_plot = df.groupby('week_start').pred.mean()
df_plot = pd.DataFrame(df_plot)
df_plot = df_plot.loc[df_plot.index < dt.datetime(2021, 1,1)]

# lineplot for political score
fig, ax = plt.subplots(figsize=(15, 10))
sns.lineplot(x = 'week_start', y = 'pred', data = df_plot, ax = ax)
sns.despine()
plt.ylabel('Predicted Polarization', fontsize = 35)
plt.xlabel('Weeks', fontsize = 35)
plt.yticks(np.arange(-0.25, -0.1, 0.05), fontsize = 30)
plt.xticks(fontsize = 30)
max_value = df_plot["pred"].max() - df_plot["pred"].max()*0.001
pandemic = pd.datetime(2020, 2, 24)
plt.axvline(pandemic, linewidth = 1, alpha = 1, color = 'red', linestyle = '--')

annotation_dict = {
    "1": (2020, 3, 9)
    , '2': (2020, 5, 4)
    , '3': (2020, 6, 15)
    , '4': (2020, 8, 10)
    , '5': (2020, 10, 12)
    , '6': (2020, 11, 2)
}
ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

for key, value in annotation_dict.items():

    week_date = dt.datetime(value[0], value[1], value[2])
    week_date = week_date - dt.timedelta(days=week_date.weekday())
    # Shift to the right the annotation
    week_date2 = dt.datetime(value[0], value[1], value[2] + 8)
    week_date2 = week_date2 - dt.timedelta(days=week_date2.weekday())

    plt.axvline(week_date,linewidth=1.5, alpha = 0.7, color='dimgrey', linestyle = '-.')
    ax.text(week_date2, max_value, key, fontsize = 25, alpha = 0.7)

save_fig(f'{path_to_figures}final/fig_2_polarization_time')