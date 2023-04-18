###########################################################
# Code to create hashtag correlations
# Author: Luca Adorni
# Date: March 2023
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

tqdm.pandas()

import sys
sys.path.append("/Users/ADORNI/Documents/graph-tool/src")

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

os.makedirs(path_to_figures_corr, exist_ok = True)

# 1. LOAD DATASET ---------------------------------------------------------------------------------



# PARAMETERS ---------
stem = True
stem_tag = np.where(stem, "_stemm","")
# If do_mentions = True, extract mentions instead of hashtags
do_mentions = False
# Set to True if we want to count just once a tweet if a hashtag appears, otherwise count the number of hashtag appearances
count_hash_app = False
# Set to True if we want to check the correlation in n. of tweets
pol_activity = False
# PARAMETER: Correlation method
method = 'spearman'
# STL Smoothing parameters
period = 2
seasonal = 7
# -----------------

def build_resid(period = 44, seasonal = 5):
    # Get the weekly-activity for each:
    # a. polarization group
    if pol_activity:
        corr_df = df.groupby(['week_start', 'polarization_bin'], as_index = False).n_tweets.sum()
        corr_df = corr_df.pivot(columns = ['polarization_bin'], index = 'week_start', values = 'n_tweets')
    else:
        corr_df = df.groupby(['week_start', 'polarization_bin'], as_index = False).scree_name.count()
        corr_df = corr_df.pivot(columns = ['polarization_bin'], index = 'week_start', values = 'scree_name')
    hashtag_df = final_df.groupby(['week_start'], as_index = True)[[k for k in hash_dict.keys()]].sum()

    corr_df = pd.concat([corr_df.loc[corr_df.index >= pd.datetime(2020, 2, 24)], hashtag_df], axis = 1)

    # Use STL to detrend/deseasonalize all our observations
    store = pd.DataFrame()
    for col in corr_df.columns:
        stl = STL(corr_df[col], period = period, seasonal = seasonal)
        res = stl.fit()
        store[col] = res.resid
    # Fix order of index
    store = store.T.reindex(['far_left', 'center_left', 'center', 'center_right', 'far_right', 
                             'anti-gov', 'immuni', 'pro-lock', 'lombardia', 
                             ]).T

    return store


# Load User/Polarization dataset
df = pd.read_pickle(f"{path_to_processed}final_df_clean.pkl.gz", compression = 'gzip')
df = df.loc[(df.week_start <= pd.datetime(2020, 12, 31))&(df.week_start >= pd.datetime(2020, 2, 24))]
df.drop(columns = 'tweet_text', inplace = True)

# Categorize user activity for their sentiment
df['main_emo'] = df[['anger', 'fear', 'joy', 'sadness']].idxmax(axis = 1)

# Load our main dataset
final_df = pd.read_pickle(f"{path_to_processed}cleaned_gsdmm_tweets{stem_tag}.pkl.gz", compression = 'gzip')
# Restrict the time
final_df = final_df.loc[(final_df.week_start <= pd.datetime(2020, 12, 31))&(final_df.week_start >= pd.datetime(2020, 2, 24))]

# Search for all hashtags
def get_hashtag(x):
    # Lowercase everything
    x = x.lower()
    # We first remove any trace of covid
    x = re.sub('(coronavirus|covid19|covid-19|covid19italia|coronavid19|pandemia|corona|virus|covid|covid_19)', '', x)  # remove hash tags
    if do_mentions:
        return re.findall("@[A-Za-z]+[A-Za-z0-9-_]+", x)
    else:
        return re.findall("#[A-Za-z]+[A-Za-z0-9-_]+",x)

# Extract only relevant hashtags
final_df['hashtag'] = final_df.tweet_text.progress_apply(lambda x: get_hashtag(x))

# Get the top hashtags
hashtags = final_df.hashtag.tolist()

# Flatten it
hashtags = [l for lis in hashtags for l in lis]
hashtags = pd.DataFrame(hashtags, columns = ['hashtags'])

# Group hashtags per topics
hash_dict = {'anti-gov': [h for h in  hashtags.hashtags.unique().tolist() if 
                            (re.search("governo|conte|speranza|pd|m5s|euro|ue|eu", h) 
                            and re.search("criminal|vergogn|dimett|irresp|merd|infam|acasa|cazzo|c4zz0", h)) 
                            or (re.search("pdiot|pidiot", h)) or (re.search("dittatura", h))],
             'pro-lock': [h for h in  hashtags.hashtags.unique().tolist() if (re.search("casa", h) 
                            and re.search("stare|sto|stiamo|resto|restare|restiamo|rimanere|rimango|rimaniamo", h) 
                            and re.search("cazzo|non|rotto|odio|accidenti|beata|nn|c4zz0|mah", h) == None) 
                            or (re.search("andra|andare", h) 
                            and re.search("bene", h) and re.search("sega|cazzo|non|accidenti|beata|nn|c4zz0|mah", h) == None)
                            or (re.search("uniti", h) and (re.search("distanti|restiamo|restare", h)))],
             'immuni': [h for h in  hashtags.hashtags.unique().tolist() if (re.search("immuni", h)
                    and re.search("immuniz|immunit", h) == None)
                    or (re.search('privacy|tracing|tracciamento', h))],
             'lombardia': [h for h in  hashtags.hashtags.unique().tolist() if re.search("lombardia|fontana|gallera", h)],
}
  

# Now flag all the tweets with at least one of those hashtags
for col, v in hash_dict.items():
    if count_hash_app:
        final_df[col] = final_df.hashtag.progress_apply(lambda x: bool(set(x) & set(v))).astype(int)
    else:
        final_df[col] = final_df.hashtag.progress_apply(lambda x: sum(el in x for el in v))

# Drop hashtag column, we do not need it anymore
final_df.drop(columns = ['hashtag'], inplace = True)

# Merge it with the dataset for our analysis/regressions
df = pd.read_pickle(f"{path_to_processed}final_df_analysis.pkl.gz", compression = 'gzip')
merge = final_df[['scree_name', 'week_start', 'anti-gov', 'pro-lock', 'immuni', 'lombardia']]
# Get the sum for user/week pair
merge = merge.groupby(['week_start', 'scree_name'], as_index = False).sum()
df = df.merge(merge, on = ['week_start', 'scree_name'], how = 'left', validate = '1:1')

# Save it
df.to_pickle(f"{path_to_processed}final_df_analysis.pkl.gz", compression = 'gzip')

# Produce an export for stata
df.rename(columns = {'anti-gov': 'anti_gov', 'pro-lock':'pro_lock'}, inplace = True)
df[['scree_name', 'prediction', 'pol_old', 'bin_old', 'orient_change_toleft', 'orient_change_toright', 'extremism_toleft', 'extremism_toright', 'center', 'center_left', 'center_right', 'far_left', 'far_right'
    , 'sentiment', 'anger', 'fear', 'joy', 'sadness', 'n_tweets', 'tot_activity', 'gender', 
    'age', 'week_start', 'dist', 'regions', 'treat', 'anti_gov', 'pro_lock', 'immuni', 'lombardia']].to_stata(f"{path_to_processed}final_df_analysis.dta.gz", compression = 'gzip'
                                           , convert_dates= {'week_start': '%tw'},
                                           version = 117)

# Build our dataset
store = build_resid(period, seasonal)
# Build a pearson correlation matrix
corr_matrix = store.corr(method = method)

# Change names of columns and rows
corr_matrix.columns = ['Far Left', 'Center Left', 'Center', 'Center Right', 'Far Right', 'Anti-Gov'
                       , 'Immuni', 'Pro-Lock', 'Lombardia']
corr_matrix.index = ['Far Left', 'Center Left', 'Center', 'Center Right', 'Far Right', 'Anti-Gov'
                       , 'Immuni', 'Pro-Lock', 'Lombardia']

# Export to LaTex the correlation table
corr_matrix.to_latex(f"{path_to_figures_corr}table_a6_corr_table.tex", header = True, index = True)

# Save it also as a pickle
corr_matrix.to_pickle(f"{path_to_figures_corr}corr_matrix.pkl.gz", compression = 'gzip')
corr_matrix.to_csv(f"{path_to_figures_corr}corr_matrix.csv.gz", compression = 'gzip', index = False)