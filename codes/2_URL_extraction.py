###########################################################
# Code to extract URLs from our Twitter dataset
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

# Setup Repository
with open("repo_info.txt", "r") as repo_info:
    path_to_repo = repo_info.readline()

path_to_data = f"{path_to_repo}data/"
path_to_raw = f"{path_to_data}raw/"
path_to_links = f"{path_to_data}links/"

# 1. Read Files -------------------------------------------------

# First step will be to read the raw scraped files

# First we import the pre-covid scrape dataset
pre_df = pd.read_feather(f'{path_to_raw}pre_covid_scrape_df')
# Then also the post-covid scrape
post_df = pd.read_csv(f'{path_to_raw}tweets_without_duplicates_regions_sentiment_demographics_topic_emotion.csv')

# And merge the two
df = pd.concat([pre_df, post_df])
assert df.shape[0] == pre_df.shape[0] + post_df.shape[0]
del pre_df, post_df

# 2. Extract URLs -----------------------------------------------

# find all urls within the tweets, gives back a list
df["url"] = df.tweet_text.apply(lambda x: re.findall("(?P<url>https?://[^\s]+)", x))

# Transform it to a set of unique URLs
url_list = [url for urls in df.url.tolist() for url in urls]
url_list = list(set(url_list))

with open(f'{path_to_links}url_list.pkl', 'wb') as f: 
    pickle.dump(url_list, f)