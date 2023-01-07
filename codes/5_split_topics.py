###########################################################
# Code to extract URLs from our Twitter dataset
# Author: Luca Adorni
# Date: September 2022
###########################################################

# 0. Setup -------------------------------------------------

#!/usr/bin/env python
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD # initialize communications
    rank = comm.Get_rank()
    size = comm.Get_size()
    mpi_use = True
except:
    print("Not running on a cluster")
    rank = 0
    size = 1
    mpi_use = False

# 1. Imports -------------------------------------------------

import numpy as np
import pandas as pd
import sys
import os
import re
import pickle
# Import libraries
import requests
import time
import sys, signal
import random

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
path_to_raw = f"{path_to_data}raw/"
path_to_links = f"{path_to_data}links/"
path_to_topic = f"{path_to_data}topic/"
path_to_results = f"{path_to_data}results/"
path_to_models = f"{path_to_data}models/"
path_to_models_top = f"{path_to_models}topic_class/"
path_to_alberto = "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0"

os.makedirs(path_to_figures, exist_ok = True)
os.makedirs(path_to_topic, exist_ok = True)
os.makedirs(path_to_results, exist_ok = True)
os.makedirs(path_to_models, exist_ok = True)
os.makedirs(path_to_models_top, exist_ok = True)

# 2. Split the dataset -------------------------------------------------

# Load the scraped dataset
scraped_df = pd.read_pickle(f'{path_to_raw}pre_covid_scrape_df_union.pkl.gz', compression='gzip')

# Load the old dataset
post_df = pd.read_csv(f'{path_to_raw}tweets_without_duplicates_regions_sentiment_demographics_topic_emotion.csv')

# Get the non-org URL
post_df = post_df.loc[post_df.is_org == 'non-org']

# Keep only the Users in post_df
scraped_df = scraped_df.loc[scraped_df.scree_name.isin(post_df.scree_name.unique().tolist())]
del post_df

# Save the cleaned dataset
scraped_df.to_pickle(f'{path_to_raw}pre_covid_scrape_df_union.pkl.gz', compression='gzip')

size = 250
step = round(scraped_df.shape[0]/size)
minimum = 0

for i in range(0,size):
    maximum = minimum + step
    if i == size - 1:
        maximum = scraped_df.shape[0]
    print(f"Going from {minimum} to {maximum}, iter {i}")
    batch = scraped_df.iloc[minimum:maximum]
    batch.to_pickle(f'{path_to_raw}batches/batch_{i}.pkl.gz', compression='gzip')
    minimum += step

size = 250
completed = []
tbd = []
failed = []

for i in range(0,size):
    # Load the old dataset
    try:
        batch = pd.read_pickle(f'{path_to_raw}batches/batch_{i}.pkl.gz', compression = 'gzip')
        if "final_pred" in batch.columns:
            print(f"Batch {i} ok")
            completed.append(batch)
        else:
            tbd.append(batch)
    except:
        failed.append(i)

assert (len(tbd)) == 0

completed = pd.concat(completed)
completed.to_pickle(f"{path_to_data}processed/pre_df5.pkl.gz", compression = 'gzip')

# # # Remove the single batches
for i in range(0, size):
    if os.path.exists(f'{path_to_raw}batches/batch_{i}.pkl.gz'): os.remove(f'{path_to_raw}batches/batch_{i}.pkl.gz')