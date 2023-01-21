###########################################################
# Code to create plot training set graphs
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

# Load our dataset
training = pd.read_pickle(f"{path_to_data}processed/training.pkl.gz", compression = 'gzip')

# define a function that saves figures
def save_fig(fig_id, tight_layout=True):
    # The path of the figures folder ./Figures/fig_id.png (fig_id is a variable that you specify 
    # when you call the function)
    path = os.path.join(os.getcwd(), "Figures", fig_id + ".png") 
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

# POLARIZATION DISTRIBUTION -----------------------
# define plot dataframe
df_plot = training.groupby(['week_start','scree_name']).final_polarization.mean()
df_plot = pd.DataFrame(df_plot)
# histogram
fig, ax = plt.subplots(figsize=(15, 10))
sns.histplot(data= df_plot, x = "final_polarization", kde = True, bins = 50, ax = ax)
sns.despine()
plt.ylabel('Week/User Count', fontsize = 25)
plt.xlabel('Polarization Score', fontsize = 25)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
plt.axvline(0, linewidth = 1, alpha = 0.9, color='r', linestyle = '--')
save_fig(f'{path_to_figures}figure_1a_training_set_hist')


# POLARIZATION OVER TIME -----------------------
df_plot = training.groupby(['week_start', 'scree_name']).final_polarization.mean().groupby('week_start').mean()
df_plot = pd.DataFrame(df_plot)
# lineplot for political score
fig, ax = plt.subplots(figsize=(15, 10))
sns.lineplot(x = 'week_start', y = 'final_polarization', data = df_plot, ax = ax)
sns.despine()
plt.ylabel('Political Score', fontsize = 25)
plt.xlabel('Weeks', fontsize = 25)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
save_fig(f'{path_to_figures}figure_1b_training_set_weekly')