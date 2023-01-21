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

# 1. BERT Results -------------------------------------------------

# Get latex code for the results
with open(f'{path_to_results}polarization_performance_reg.pkl', 'rb') as handle:
    results_test = pickle.load(handle)

def get_results(results_dict):
    final_results = []
    for model, res in results_test.items():
        lr = re.findall("lr_(.*?)_", model)[0]
        batch = re.findall("_batch_(.*?)_", model)[0]
        perc = re.findall("perc_(.*?)_", model)[0]
        len = re.findall("_len_(.*?)$", model)[0]
        type = re.findall("_batch_[0-9]*_(.*?)_", model)[0]

        final_results.append([type, str(f"_{lr}"), batch, perc, len, res["MAE"], res["MSE"], res["R2"]])
    return final_results

final_res = get_results(results_test)

table = tabulate(final_res, tablefmt = 'latex', headers = ['Dataset', 'LR', 'Batch', 'Threshold', 'Len', "MAE", "MSE", "R2"], floatfmt=".4f")

with open(f'{path_to_tables}table_4_bert_res.tex', 'w') as f:
    f.write(table)