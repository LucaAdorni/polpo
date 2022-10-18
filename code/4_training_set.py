###########################################################
# Code to create a training dataset
# Author: Luca Adorni
# Date: September 2022
###########################################################

# 0. Setup -------------------------------------------------

import numpy as np
import pandas as pd
import os
import re
import dill
import pickle
from tqdm import tqdm

pd.options.display.max_columns = 200
pd.options.display.max_rows = 1000
pd.set_option('max_info_columns', 200)
pd.set_option('expand_frame_repr', False)
pd.set_option('expand_frame_repr', True)
pd.set_option('max_colwidth',1000)
pd.set_option('display.width',None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Setup Repository
with open("repo_info.txt", "r") as repo_info:
    path_to_repo = repo_info.readline()

path_to_data = f"{path_to_repo}data/"
path_to_raw = f"{path_to_data}raw/"
path_to_links = f"{path_to_data}links/"
path_to_processed = f"{path_to_data}processed/"

os.makedirs(path_to_processed, exist_ok=True)

# 1. Merge Pre and Post Dataset -------------------------------------------------

# We first import and merge our two main raw datasets
post_df = pd.read_csv(f"{path_to_raw}tweets_without_duplicates_regions_sentiment_demographics_topic_emotion.csv")
pre_df = pd.read_feather(f'{path_to_raw}pre_covid_scrape_df')

# Get only user info
user_info = post_df[['user_description', 'scree_name', 'regions', 'age', 'gender', 'is_org']]
# drop duplicates
user_info.drop_duplicates(subset = 'scree_name',inplace = True)
assert user_info.shape[0] == post_df.scree_name.nunique()
# Now merge it with our pre-covid scrape
pre_df = pre_df.merge(user_info, on = 'scree_name', how = 'outer', validate = 'm:1', indicator = True)

# we drop unmerged values
pre_df = pre_df.loc[pre_df._merge == 'both']
pre_df.drop(columns = '_merge', inplace = True)

# Now we merge the two datasets and drop the data we do not need
df = pd.concat([pre_df, post_df])
del user_info, pre_df, post_df

# Finally we keep only users which are not organizations
print(f"Length pre-organization drop: {df.shape[0]}")
df = df.loc[df.is_org == False] #MISTAKE HERE
print(f"Length post-organization drop: {df.shape[0]}")

df.to_pickle(f'{path_to_processed}final_df.pkl.gz', compression='gzip')

# 2. Distant Supervision -------------------------------------------------
