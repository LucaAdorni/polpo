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
size = 50
store = []
failed = []
check_merge = True
# Then try to load scraped datasets - if we find none of them, then we haven't iteratively scraped
try:
    for i in range(0, size):
        try:
            scraped_i = pd.read_pickle(f'{path_to_raw}pre_covid_scrape_df_{i}.pkl.gz', compression='gzip')
            store.append(scraped_i)
        except:
            print(f"Failed list: {i}")
            failed.append(i)

    if len(failed) == size:
        check_merge = False
except:
    print("No past scrape")
    store = []
    check_merge = False
try:
    scraped_df = pd.read_pickle(f'{path_to_raw}pre_covid_scrape_df_union.pkl.gz', compression='gzip')
    print(f"Past merged scrape exist: {scraped_df.shape[0]}")
    print(f"Unique users: {scraped_df.scree_name.nunique()}")
except:
    print("No past merged scrape")
    scraped_df = pd.DataFrame()
store.append(scraped_df)
scraped_df = pd.concat(store)
del store
if check_merge:
    scraped_df.drop_duplicates(inplace = True)
    scraped_df.to_pickle(f'{path_to_raw}pre_covid_scrape_df_union.pkl.gz', compression='gzip')
    # Now remove the single files to avoid cluttering
    for i in range(0, size):
        if os.path.exists(f'{path_to_raw}pre_covid_scrape_df_{i}.pkl.gz'): os.remove(f'{path_to_raw}pre_covid_scrape_df_{i}.pkl.gz')


# find all urls within the tweets, gives back a list
scraped_df["url"] = scraped_df.tweet_text.apply(lambda x: re.findall("(?P<url>https?://[^\s]+)", x))

# Transform it to a set of unique URLs
url_list = url_list + [url for urls in scraped_df.url.tolist() for url in urls]
url_list = list(set(url_list))

with open(f'{path_to_links}url_list.pkl', 'wb') as f: 
    pickle.dump(url_list, f)
    print("Saved URL list")