###########################################################
# Code to plot the training set sizes
# Author: Luca Adorni
# Date: January 2023
###########################################################

# 0. Setup -------------------------------------------------

import numpy as np
import pandas as pd
import os
import sys
import re
import pickle
import random
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import dill

try:
    # Setup Repository
    with open("repo_info.txt", "r") as repo_info:
        path_to_repo = repo_info.readline()
except:
    path_to_repo = f"{os.getcwd()}/polpo/"
    sys.path.append(f"{os.getcwd()}/.local/bin") # import the temporary path where the server installed the module


print(path_to_repo)

path_to_data = f"{path_to_repo}data/"
path_to_figures = f"{path_to_repo}figures/"
path_to_tables = f"{path_to_repo}tables/"
path_to_raw = f"{path_to_data}raw/"
path_to_links = f"{path_to_data}links/"
path_to_topic = f"{path_to_data}topic/"
path_to_results = f"{path_to_data}results/"
path_to_models = f"{path_to_data}models/"
path_to_models_pol = f"{path_to_models}ML/"
path_to_processed = f"{path_to_data}processed/"

os.makedirs(path_to_figures, exist_ok = True)
os.makedirs(path_to_tables, exist_ok = True)
os.makedirs(path_to_topic, exist_ok = True)
os.makedirs(path_to_results, exist_ok = True)
os.makedirs(path_to_models, exist_ok = True)
os.makedirs(path_to_models_pol, exist_ok = True)

# 1. Import the datasets -------------------------------------------------

infos = []
tag_list = ['train', 'val', 'test']

for end in ['', '_merged']:
    for tag in tag_list:
        df = pd.read_pickle(f"{path_to_processed}df_{tag}{end}.pkl.gz", compression = 'gzip')
        size = df.shape[0]
        users = df.scree_name.nunique()
        weeks = df.week_start.nunique()
        if end == "":
            infos.append([tag, "individual", size, users, weeks])
        else:
            infos.append([tag, "merged", size, users, weeks])

table = tabulate(infos, tablefmt = 'latex', headers = ['Dataset', 'Type', 'Size', 'Users', 'Weeks'], floatfmt="5.0f")

with open(f'{path_to_tables}table_1_training_set.tex', 'w') as f:
    f.write(table)