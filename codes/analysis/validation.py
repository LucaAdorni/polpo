###########################################################
# Code to create validation graphs
# Author: Luca Adorni
# Date: April 2023
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
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.seasonal import STL
import networkx as nx
from tqdm import tqdm
import sys

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
path_to_figure_odds = f"{path_to_figures}logit_res/"
path_to_tables = f"{path_to_repo}tables/"


path_to_figures_corr = f"{path_to_figures}corr_heat/"
path_to_figures_final = f"{path_to_figures}final/"


# 1. LOAD DATASET ------------------------------------------------------------


df = pd.read_pickle(f"{path_to_processed}final_df_analysis.pkl.gz", compression = 'gzip')


from binsreg import binsregselect, binsreg, binsqreg, binsglm, binstest, binspwc


help(binsreg)


binsregselect(y = 'anger', x = 'prediction', data = df,  bins = (3, 0), nbins = 40)


binsreg(y = df.anger, x = df.prediction)