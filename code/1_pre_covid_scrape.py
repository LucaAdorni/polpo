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
except:
    print("Not running on a cluster")
    rank = 0

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

print("Process - Rank: %d -----------------------------------------------------"%comm.rank)

if rank == 0:
    # First step will be to read the raw scraped files
    #post_df = pd.read_csv(f'{path_to_raw}tweets_without_duplicates_regions_sentiment_demographics_topic_emotion.csv')

    # and get the list of users we want to scrape
    #user_list = post_df.scree_name.tolist()
    #user_list = list(set(user_list))
    with open(f'{path_to_raw}user_list.pkl', 'rb') as f: 
        user_list = pickle.load(f)
    #del post_df
    # Add a check to see if the module is working
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper('from:{} since:{} until:{}'.format("BattagliaInfo", "2020-04-01", "2020-05-01")).get_items()):
        if i>1:
            break
        print(tweet.id)
    try:
        user_list = split_list(user_list)
    except:
        print("")
else:
    user_list = None

# Now we send our splitted list of proxies to our nodes
try:
    user_list = comm.scatter(user_list, root = 0)
except:
    print("")

# SCRAPING --------------------------------------------------------------------------------------

try :
    scraped_df = pd.read_pickle(f'{path_to_raw}pre_covid_scrape_df_{rank}.pkl.gz', compression='gzip')
    scraped_df.drop_duplicates(inplace = True)
except :
    scraped_df = pd.DataFrame() # initiate our empty list of tweets


try :
    scraped_df2 = pd.read_pickle(f'{path_to_raw}pre_covid_scrape_df_2.pkl.gz', compression='gzip')
    scraped_df2.drop_duplicates(inplace = True)
except :
    scraped_df2 = pd.DataFrame() # initiate our empty list of tweets    

try :
    scraped_df1 = pd.read_pickle(f'{path_to_raw}pre_covid_scrape_df_1.pkl.gz', compression='gzip')
    scraped_df1.drop_duplicates(inplace = True)
except :
    scraped_df1 = pd.DataFrame() # initiate our empty list of tweets    


# PARAMETERS -----------------------------------------------------------------------------------------------------

start_date = '2020-01-01'
end_date = '2020-02-29'

try:
    # We start from the users we already have
    print(len(user_list))
    user_list = list(set(user_list) - set(scraped_df.scree_name.unique()))
    user_list = list(set(user_list) - set(scraped_df2.scree_name.unique()))
    print(len(user_list))
    minimum = 0
except:
    minimum = 0 # beginning of our list

maximum = len(user_list) # end of our list
steps = 1000 # how many urls per batch
range_list = list(np.arange(minimum,maximum,steps))

iter_count = 0

tweet_list_scraped = []
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
    if iter_count % 20000 == 0 or max_r == maximum:
        # We first flatten out our list of tweets
        flat_list = [t for sublist in tweet_list_scraped for item in sublist for t in item]
        # In case we have only errors, add a fake tweet
        flat_list.append([0, "-", "-", "-", "-", "-"])
        # Creating a dataframe from the tweets list above 
        new_batch = pd.DataFrame(flat_list, columns=['tweet_ids', 'tweet_text', 'locations', 'dates', 'scree_name', 'user_id'])
        # drop errors
        new_batch = new_batch.loc[(new_batch.tweet_ids != 'Error')&(new_batch.tweet_ids != 0)]
        scraped_df = pd.concat([scraped_df2, new_batch])
        scraped_df.reset_index(inplace = True, drop = True)
        scraped_df.to_pickle(f'{path_to_raw}pre_covid_scrape_df_{rank}.pkl.gz', compression='gzip')
        print(f"Output Saved - {min_r} to {max_r} out of {maximum}")