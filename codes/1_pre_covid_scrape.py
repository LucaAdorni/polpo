###########################################################
# Code to scrape tweets pre-covid from our dataset
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

import numpy as np
import pandas as pd
import os
import re
import pickle
import sys
import snscrape.modules.twitter as sntwitter
from concurrent.futures import ThreadPoolExecutor
import time

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

# 1. Define Functions and Parameters ------------------------------------------------------------------------------

def split_list(list_split):
    """
    Function to split a list to be then sent to our various nodes
    """
    send_list = []
    list_len = len(list_split)//size
    for i in range(0, size):
        if i == size-1:
            send_list.append(list_split[i*list_len:len(list_split)])
        else:
            send_list.append(list_split[i*list_len:i*list_len + list_len])

    return send_list

def scrape_users(username):
    """ scrape all tweets from a username"""
    tweet_list = []
    # Using TwitterSearchScraper to scrape data and append tweets to list
    try:
        for i,tweet in enumerate(sntwitter.TwitterSearchScraper('from:{} since:{} until:{}'.format(username, start_date, end_date)).get_items()):
            if i>1000:
                break
            tweet_list.append([tweet.id, tweet.content, tweet.user.location, tweet.date, tweet.user.username, tweet.user.id])
    except:
        tweet_list.append(['Error', username])
    return tweet_list

def iterate_scraping(min_r, max_r, user_list, pool_n = 10000):    
    """ Function to thread pool our scraping function """
    start = time.time()
    # create our list of tweets
    tweets_list1 = []
    # create user list
    batch_user = user_list[min_r:max_r]
    # start the thread pool
    with ThreadPoolExecutor(pool_n) as executor:
        # execute tasks concurrently and process results in order
        for result in executor.map(scrape_users, batch_user):
            # retrieve the result
            tweets_list1.append(result)
        # shutdown the executor
        executor.shutdown()
    end = time.time()
    print("Elapsed Time: {}".format(end - start))
    return tweets_list1

# 2. Read Files --------------------------------------------------------------------------------------------------

print("Process - Rank: %d -----------------------------------------------------"%rank)

if rank == 0:
    # Load any previous list of scraped users
    with open(f"{path_to_raw}scraped_users_union.pkl", "rb") as fp:   # Unpickling
        scraped_users = pickle.load(fp)
        try: 
            scraped_users = scraped_users.tolist()
        except:
            print("")
        print(len(scraped_users))
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
        scraped_users = scraped_users + scraped_df.scree_name.unique().tolist()
        scraped_users = list(set(scraped_users))
        del scraped_df
        print(len(scraped_users))
        with open(f"{path_to_raw}scraped_users_union.pkl", "wb") as fp:   #Pickling
            pickle.dump(scraped_users, fp)

        # Now remove the single files to avoid cluttering
        for i in range(0, size):
            if os.path.exists(f'{path_to_raw}pre_covid_scrape_df_{i}.pkl.gz'): os.remove(f'{path_to_raw}pre_covid_scrape_df_{i}.pkl.gz')

if rank == 0:
    try:
        with open(f'{path_to_raw}user_list.pkl', 'rb') as f: 
            user_list = pickle.load(f)
    except:
        # First step will be to read the raw scraped files
        post_df = pd.read_csv(f'{path_to_raw}tweets_without_duplicates_regions_sentiment_demographics_topic_emotion.csv')

        # and get the list of users we want to scrape
        user_list = post_df.scree_name.tolist()
        user_list = list(set(user_list))
        del post_df
    with open(f'{path_to_repo}polpo_log.txt', 'w') as f:
        f.write('trying scrape')
    
    # We start from the users we already have
    print(len(user_list))
    user_list = list(set(user_list) - set(scraped_users))
    print(len(user_list))
    del scrape_users
    with open(f'{path_to_repo}polpo_log.txt', 'w') as f:
        f.write(f'scraped_tweets = missing users: {len(user_list)}')

    if mpi_use:
        user_list = split_list(user_list)
else:
    user_list = None

# Now we send our splitted list of proxies to our nodes
if mpi_use:
    user_list = comm.scatter(user_list, root = 0)

# SCRAPING --------------------------------------------------------------------------------------

# PARAMETERS ###############

start_date = '2020-01-01'
end_date = '2020-02-29'

# Initialize an empty dataset
scraped_df = pd.DataFrame()

minimum = 0
maximum = len(user_list) # end of our list
steps = 1000 # how many urls per batch
range_list = list(np.arange(minimum,maximum,steps))

iter_count = 0

tweet_list_scraped = []
missing_users = []

for i in range(len(range_list)):
    min_r = range_list[i]
    try: 
        max_r = range_list[i+1]
    except: 
        max_r = maximum # if we are out of range we use the list up to its maximum
    print("Range: {} to {}".format(min_r, max_r))
    batch_list = iterate_scraping(min_r, max_r, user_list, pool_n = 100) # our iterating thread pool to request urls
    tweet_list_scraped.append(batch_list) # we merge the results we already have with the ones of the current batch
    print("Batch ended")
    iter_count += steps
    if iter_count % 4000 == 0 or max_r == maximum:
        # We first flatten out our list of tweets
        flat_list = [t for sublist in tweet_list_scraped for item in sublist for t in item]
        # In case we have only errors, add a fake tweet
        flat_list.append([0, "-", "-", "-", "-", "-"])
        # Creating a dataframe from the tweets list above 
        new_batch = pd.DataFrame(flat_list, columns=['tweet_ids', 'tweet_text', 'locations', 'dates', 'scree_name', 'user_id'])
        # drop errors
        new_batch = new_batch.loc[(new_batch.tweet_ids != 'Error')&(new_batch.tweet_ids != 0)]
        scraped_df = pd.concat([scraped_df, new_batch])
        scraped_df.reset_index(inplace = True, drop = True)
        scraped_df.to_pickle(f'{path_to_raw}pre_covid_scrape_df_{rank}.pkl.gz', compression='gzip')
        print(f"Output Saved - {min_r} to {max_r} out of {maximum}")
        print(f"N. of rows: {scraped_df.shape}")
        # Get what users we are missing from our batches
        missing_users = missing_users + list(set(user_list) - set(scraped_df.scree_name.unique().tolist()))

len_miss = len(missing_users)

while len_miss > 1000:

    maximum = len(missing_users) # end of our list
    steps = 1000 # how many urls per batch
    range_list = list(np.arange(minimum,maximum,steps))

    iter_count = 0

    tweet_list_scraped = []
    missing = []

    for i in range(len(range_list)):
        min_r = range_list[i]
        try: 
            max_r = range_list[i+1]
        except: 
            max_r = maximum # if we are out of range we use the list up to its maximum
        print("Range: {} to {}".format(min_r, max_r))
        batch_list = iterate_scraping(min_r, max_r, missing_users, pool_n = 100) # our iterating thread pool to request urls
        tweet_list_scraped.append(batch_list) # we merge the results we already have with the ones of the current batch
        print("Batch ended")
        iter_count += steps
        if iter_count % 4000 == 0 or max_r == maximum:
            # We first flatten out our list of tweets
            flat_list = [t for sublist in tweet_list_scraped for item in sublist for t in item]
            # In case we have only errors, add a fake tweet
            flat_list.append([0, "-", "-", "-", "-", "-"])
            # Creating a dataframe from the tweets list above 
            new_batch = pd.DataFrame(flat_list, columns=['tweet_ids', 'tweet_text', 'locations', 'dates', 'scree_name', 'user_id'])
            # drop errors
            new_batch = new_batch.loc[(new_batch.tweet_ids != 'Error')&(new_batch.tweet_ids != 0)]
            scraped_df = pd.concat([scraped_df, new_batch])
            scraped_df.reset_index(inplace = True, drop = True)
            scraped_df.to_pickle(f'{path_to_raw}pre_covid_scrape_df_{rank}.pkl.gz', compression='gzip')
            print(f"Output Saved - {min_r} to {max_r} out of {maximum}")
            print(f"N. of rows: {scraped_df.shape}")
            # Get what users we are missing from our batches
            missing = missing + list(set(missing_users) - set(scraped_df.scree_name.unique().tolist()))
    len_miss = len(missing)
    missing_users = missing.copy()