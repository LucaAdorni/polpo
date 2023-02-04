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
    path = os.path.join(os.getcwd(), "Figures", fig_id + ".pdf") 
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='pdf', dpi=300)

# 1. Load Dataset ----------------------------------------------------


# Load our dataset
df = pd.read_pickle(f"{path_to_processed}final_df.pkl.gz", compression = 'gzip')

# Restrict to users who have posted at least once in the pre-covid period
df['min_date'] = df.groupby('scree_name').week_start.transform('min')
df['max_date'] = df.groupby('scree_name').week_start.transform('max')
df['check'] = (df.min_date <= pd.datetime(2020, 2, 18))& (df.max_date >= pd.datetime(2020, 2, 18))

df_plot = df.loc[df.check].groupby('week_start').prediction.mean()
df_plot = pd.DataFrame(df_plot)
df_plot = df_plot.loc[df_plot.index < dt.datetime(2021, 1,1)]

# lineplot for political score
fig, ax = plt.subplots(figsize=(15, 10))
sns.lineplot(x = 'week_start', y = 'prediction', data = df_plot, ax = ax)
sns.despine()
plt.ylabel('Political Score')
plt.xlabel('Weeks')

annotation_dict = {
      'First COVID-19 Cases': (2020, 2, 21)
      , 'DPCM #IoRestoaCasa': (2020, 3, 9)
      , 'End of Lockdown': (2020, 5, 6)
      , 'Immuni is released': (2020, 6, 15)
      , 'Nightclub closure': (2020, 8, 17)

  }

iter_value = 0
max_value = df_plot["prediction"].max() - df_plot["prediction"].max()*0.001
for key, value in annotation_dict.items():
 
    week_date = dt.datetime(value[0], value[1], value[2])
    week_date = week_date - dt.timedelta(days=week_date.weekday())
    title_ann = key

    plt.axvline(week_date,linewidth=1, alpha = 1, color='black')
    ax.annotate(title_ann, xy = (week_date, max_value*0.9999), xytext=(-50, 15 + iter_value), 
                    textcoords='offset points', xycoords = 'data',
                    bbox=dict(boxstyle="square", fc="white", ec="gray"), arrowprops=dict(arrowstyle='->',
                            connectionstyle="arc3"), fontsize = 18)
    iter_value += -35
    

save_fig(f'{path_to_figures}figure_2_polarization_time')