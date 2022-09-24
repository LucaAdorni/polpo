###########################################################
# Code to scrape tweets pre-covid from our dataset
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
import snscrape.modules.twitter as sntwitter
from concurrent.futures import ThreadPoolExecutor
from pytimedinput import timedInput
import time

# Setup Repository
with open("repo_info.txt", "r") as repo_info:
    path_to_repo = repo_info.readline()

path_to_data = f"{path_to_repo}data/"
path_to_raw = f"{path_to_data}raw/"
path_to_links = f"{path_to_data}links/"

# 1. Read Files --------------------------------------------------------------------------------------------------

# First step will be to read the raw scraped files
post_df = pd.read_csv(f'{path_to_raw}tweets_without_duplicates_regions_sentiment_demographics_topic_emotion.csv')

# and get the list of users we want to scrape
user_list = post_df.scree_name.tolist()
user_list = list(set(user_list))

# 2. Define Functions and Parameters ------------------------------------------------------------------------------

def scrape_users(username):
    """ scrape all tweets from a username"""
    tweet_list = []
    # Using TwitterSearchScraper to scrape data and append tweets to list
    try:
        for i,tweet in enumerate(sntwitter.TwitterSearchScraper('from:{} since:{} until:{}'.format(username, start_date, end_date)).get_items()):
            if i>10000:
                break
            tweet_list.append([tweet.id, tweet.content, tweet.user.location, tweet.date, tweet.user.username, tweet.user.id])
    except:
        tweet_list.append(['Error', username])
    return tweet_list


def iterate_scraping(min_r, max_r, user_list):    
    """ Function to thread pool our scraping function """
    start = time.time()
    # create our list of tweets
    tweets_list1 = []
    # create user list
    batch_user = user_list[min_r:max_r]
    # start the thread pool
    with ThreadPoolExecutor(100) as executor:
        # execute tasks concurrently and process results in order
        for result in executor.map(scrape_users, batch_user):
            # retrieve the result
            tweets_list1.append(result)
        # shutdown the executor
        executor.shutdown()
    end = time.time()
    print("Elapsed Time: {}".format(end - start))
    return tweets_list1

# SCRAPING --------------------------------------------------------------------------------------

try :
    scraped_df = pd.read_feather(f'{path_to_raw}pre_covid_scrape_df')
except :
    scraped_df = pd.DataFrame() # initiate our empty list of tweets

# PARAMETERS -----------------------------------------------------------------------------------------------------

start_date = '2020-01-01'
end_date = '2020-02-29'

try:
    # We start from the users we already have
    minimum = scraped_df.scree_name.nunique()
except:
    minimum = 0 # beginning of our list

maximum = len(user_list) # end of our list
steps = 1000 # how many urls per batch
range_list = list(np.arange(minimum,maximum,steps))

iter_count = 0
timeout_loop = 5

timedOut = True
tweet_list_scraped = []
for i in range(len(range_list)):
    userText, timedOut = timedInput("Please, do enter something to stop for loop: ", timeout = timeout_loop)
    timeout_loop = 5
    if timedOut == True:
        min_r = range_list[i]
        try: 
            max_r = range_list[i+1]
        except: 
            max_r = maximum # if we are out of range we use the list up to its maximum
        print("Range: {} to {}".format(min_r, max_r))
        batch_list = iterate_scraping(min_r, max_r, user_list) # our iterating thread pool to request urls
        tweet_list_scraped.append(batch_list) # we merge the results we already have with the ones of the current batch
        print("Batch ended")
        iter_count += steps
        if iter_count % 2000 == 0 or max_r == maximum:
            # We first flatten out our list of tweets
            flat_list = [t for sublist in tweet_list_scraped for item in sublist for t in item]
            # Creating a dataframe from the tweets list above 
            new_batch = pd.DataFrame(flat_list, columns=['tweet_ids', 'tweet_text', 'locations', 'dates', 'scree_name', 'user_id'])
            # drop errors
            new_batch = new_batch.loc[new_batch.tweet_ids != 'Error']
            scraped_df = pd.concat([scraped_df, new_batch])
            scraped_df.reset_index(inplace = True, drop = True)
            scraped_df.to_feather(f'{path_to_raw}pre_covid_scrape_df')
            print("Output Saved")
            timeout_loop = 30
    else:
        break