###########################################################
# Code to plot the results for polarization
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

# 1. ML Results -------------------------------------------------

# Get latex code for the results
with open(f'{path_to_results}test_results.pickle', 'rb') as handle:
    ml = pickle.load(handle)

# Get latex code for the results
with open(f'{path_to_results}val_results_tuned.pickle', 'rb') as handle:
    ml_tuned = pickle.load(handle)

method_list = {'frequency': 'TF', 'onehot': 'BIN','tf_idf': "TF-IDF"}
model_list = {'lasso': "Lasso", 'rand_for': "Random Forest", 'xgboost': "XGBoost", 'catboost': "CatBoost"}

def get_results(results_dict, tuned = "No"):
    final_results = []
    for estimator in model_list.keys():
        for method in method_list.keys():
            if estimator not in list(results_dict[method].keys()):
                final_results.append([model_list[estimator], method_list[method], tuned, np.nan, np.nan, np.nan])
            else:    
                final_results.append([model_list[estimator], method_list[method], tuned, results_dict[method][estimator]["MAE"], results_dict[method][estimator]["MSE"], results_dict[method][estimator]["R2"]])
    return final_results

res_ml = get_results(ml)
res_ml_tuned = get_results(ml_tuned, tuned = "Yes")

final_res = res_ml + res_ml_tuned

table = tabulate(final_res, tablefmt = 'latex', headers = ['Model', 'Method', 'Tuning', 'MAE', "MSE", "R2"], floatfmt=".4f")

with open(f'{path_to_tables}table_4_ml_baseline.tex', 'w') as f:
    f.write(table)