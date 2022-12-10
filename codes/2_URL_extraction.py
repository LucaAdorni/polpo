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
import pickle
import sys

try:
    # Setup Repository
    with open("repo_info.txt", "r") as repo_info:
        path_to_repo = repo_info.readline()
except:
    path_to_repo = f"{os.getcwd()}/polpo/"
    sys.path.append(f"{os.getcwd()}/.local/bin") # import the temporary path where the server installed the module

path_to_data = f"{path_to_repo}data/"
path_to_raw = f"{path_to_data}raw/"
path_to_links = f"{path_to_data}links/"

# 1. Extract URLs from COVID dataset -------------------------------------------------

try:
    with open(f'{path_to_links}url_list.pkl', 'rb') as f: 
        url_list = pickle.load(f)
        print(f"URLs in dataset: {len(url_list)}")
except:
    # First step will be to read the raw scraped files
    post_df = pd.read_csv(f'{path_to_raw}tweets_without_duplicates_regions_sentiment_demographics_topic_emotion.csv')

    # find all urls within the tweets, gives back a list
    post_df["url"] = post_df.tweet_text.apply(lambda x: re.findall("(?P<url>https?://[^\s]+)", x))

    # Transform it to a set of unique URLs
    url_list = [url for urls in post_df.url.tolist() for url in urls]
    url_list = list(set(url_list))

    # Save them as a pickle
    with open(f'{path_to_links}url_list.pkl', 'wb') as f: 
        pickle.dump(url_list, f)

    del post_df

# 2. Loop and read all the pre-scraping files -------------------------------------------------

# First step will be to read the raw scraped files
for rank in range(0,10):
    print(rank)
    pre_df = pd.read_pickle(f'{path_to_raw}pre_covid_scrape_df_{rank}.pkl.gz', compression='gzip')

    # find all urls within the tweets, gives back a list
    pre_df["url"] = pre_df.tweet_text.apply(lambda x: re.findall("(?P<url>https?://[^\s]+)", x))

    # Transform it to a set of unique URLs
    url_list = url_list + [url for urls in pre_df.url.tolist() for url in urls]
    url_list = list(set(url_list))

with open(f'{path_to_links}url_list.pkl', 'wb') as f: 
    pickle.dump(url_list, f)
    print("Saved URL list")