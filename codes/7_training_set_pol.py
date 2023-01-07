###########################################################
# Code to create a training dataset
# Author: Luca Adorni
# Date: January 2023
###########################################################

# 0. Setup -------------------------------------------------

import numpy as np
import pandas as pd
import os
import re
import dill
import pickle
from tqdm import tqdm
import sys
import tldextract
from collections import Counter

tqdm.pandas()

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

# 1. Get the Unshortened links -------------------------------------------------

# We need to first get back all the unshortened links
size = 100
failed = []
store = {}
# To ease computation, we have split our URLs into n-size lists/dictionaries (one per machine)
# Iteratively load all of them
try:
    for i in range(0, size):
        try:
            with open(f'{path_to_links}url_dictionary_{i}.pkl', 'rb') as f: 
                url_dict_i = pickle.load(f)
            store = {**store, **url_dict_i}
        except:
            print(f"Failed list: {i}")
            failed.append(i)
    # If we have none of them (or some error occurred) - signal that we do not have them
    if len(failed) == size:
        check_merge = False
    else:
        check_merge = True
except:
    print("No past scrape")
    store = {}
    check_merge = False
# If we have pre-existing URL-dictionaries for each CPU, need to merge with previous results
# And then re-create new lists of TO-DO URLs
if check_merge:
    # Load (if it exist) the previous dictionary of unshortened URLs
    try:
        with open(f'{path_to_links}url_dictionary.pkl', 'rb') as f: 
            url_dict = pickle.load(f)
        print("URL dictionary loaded")
    except:
        url_dict = {}
        print("New URL dictionary initiated")
    print(f"Pre-existing unshortened list of URLs length: {len(url_dict)}")
    # If we have some new scrape, merge it with the previous
    print(len(store))
    url_dict = {**store, **url_dict}
    del store
    print(len(url_dict))
    with open(f'{path_to_links}url_dictionary.pkl', 'wb') as f: 
            pickle.dump(url_dict, f)
            print("saved dictionary")
        
    # Now remove all those previous lists/dictionaries
    for i in range(0, size):
        if os.path.exists(f'{path_to_links}url_dictionary_{i}.pkl'): os.remove(f'{path_to_links}url_dictionary_{i}.pkl')
        if os.path.exists(f'{path_to_links}url_list_{i}.pkl'): os.remove(f'{path_to_links}url_list_{i}.pkl')
    # Check which URLs we still need to unshorten
    print("Loading the URL lists")

# 2. Define the functions to compute polarization -------------------------------------------------


# find all urls within the tweets, gives back a list
def find_urls(df):
    """
    Function to iteratively find all URLs within tweets
    """
    df["url"] = df.tweet_text.progress_apply(lambda x: re.findall("(?P<url>https?://[^\s]+)", x))

def find_full_link(df):
    """
    Function to match the unshortened links iteratively over our full dataset
    """
    # create a column of empty lists, to be filled with our full urls
    df["full_url"] = np.empty((len(df), 0)).tolist()

    def iterate_full_link(x, results):
        """
        Function to unshorten a list of URLs
        x: our list of urls
        results: our dictionary of unshortened urls
        """
        for url in x.url:
            try: x.full_url.append(results[url]) # append the results
            except: next

    df.progress_apply(lambda x: iterate_full_link(x, url_dict), axis = 1)
    # Drop the URL column since we do not need it anymore
    df.drop(columns = 'url', inplace = True)

def extract_domains(df):
    """
    Function to extract all the domains from our unshortened links
    """
    # create a column of empty lists, to be filled with our full urls
    df["domain"] = np.empty((len(df), 0)).tolist()

    def extract(url):
        """
        Function to extract the domain from each URL
        """
        uri = tldextract.extract(url)
        uri = '.'.join(uri[1:3])
        return uri

    def iterate_extract(x):
        """
        Function to extract the domain from our list of urls
        x: our list of urls
        """
        for url in x.full_url:
            try: x.domain.append(extract(url)) # append the results
            except: next

    df.progress_apply(lambda x: iterate_extract(x), axis = 1)
    # Drop the full-url column since we do not need it anymore
    df.drop(columns = 'full_url', inplace = True)

# Load the domain alignments -----------------------------------------------------------

link_labels = pd.read_csv(f'{path_to_data}domain_alignment/italian_labels.csv')
leftdf = link_labels[link_labels["polarity_cont"] <0]
rightdf = link_labels[link_labels["polarity_cont"] >0]

# Define a set of all the valid domains
valid_domains_lr = list(set(link_labels.domain))

# Generate counts
left_doms_count=None
right_doms_count=None

def clean_domains(df):
    """
    Clean our dataset from all the non-political domains (i.e. what we will not use for distant supervision)
    """
    # create a column of empty lists, to be filled with our cleaned domains
    df["domain_clean"] = np.empty((len(df), 0)).tolist()

    def clean_lr(links):
        global left_doms_count
        global right_doms_count
        #links = [extract(x) for x in links]
        links = [x for x in links if x in valid_domains_lr]

        left_doms = [x for x in links if x in leftdf['domain'].tolist()]
        right_doms = [x for x in links if x in rightdf['domain'].tolist()]
        if left_doms_count is None:
            left_doms_count=Counter(left_doms)
        else:
            left_doms_count+=Counter(left_doms)

        if right_doms_count is None:
            right_doms_count=Counter(right_doms)
        else:
            right_doms_count+=Counter(right_doms)

        if len(links)>0:
            return links
        else:
            return list()

    df['domain_clean']=df['domain'].progress_apply(lambda x: clean_lr(x))
    # Drop the domain column since we do not need it anymore
    df.drop(columns = 'domain', inplace = True)


def df_polarization(df):
    """
    Get the polarization of our dataset
    """
    # create a column of empty lists, to be filled with our full urls
    df["polarization"] = np.empty((len(df), 0)).tolist()

    def get_polarization(domain_clean):
        """
        Function to get the domain value
        """
        if domain_clean in link_labels['domain'].tolist():
            polarization = float(link_labels[link_labels.domain == domain_clean]['polarity_cont'])
        return polarization

    def iterate_polarization(x):
        """
        Function to extract the domain from our list of urls
        x: our list of urls
        """
        
        for domain in x.domain_clean:
            try: x.polarization.append(get_polarization(domain)) # append the results
            except: next

    df.progress_apply(lambda x: iterate_polarization(x), axis = 1)
    
    # count the number of clean domains per observation
    df["domain_count"] = df.polarization.progress_apply(lambda x: len(x))


def tot_polarization(df):
    """
    Compute the overall polarization of a tweet in case of multiple links
    """
    def polarization_sum(x):
        """
        Function to get the total polarization of a tweet
        x: our list of polarization
        """
        final_sum = 0
        for polarization in x:
            final_sum += polarization
        return final_sum

    df['polarization_sum'] = df.polarization.progress_apply(lambda x: polarization_sum(x))
    # Drop the polarization column since we do not need it anymore
    df.drop(columns = 'polarization', inplace = True)

def get_weekly_pol(df):
    """ Get the target variable - weekly polarization"""
    def clean_df(df):
        """ Convert datetimes and clean useless tweets"""
        # we convert our dates to a datetime object
        df.dates = pd.to_datetime(df.dates)
        # we create a subset with only tweets with at least one domain
        df_clean = df[df.domain_count > 0]
        return df_clean

    def get_weekly_sum(df):
        """
        Function to get the weekly polarization of our tweets
        """
        # We generate a matrix with the sum of weekly observations
        sum_tweet = df.groupby([pd.Grouper(key="dates", freq=frequency), "scree_name"]).sum()
        sum_tweet.reset_index(inplace=True)
        sum_tweet["final_polarization"] = sum_tweet["polarization_sum"] / sum_tweet["domain_count"]
        sum_tweet['week_start'] = pd.to_datetime(sum_tweet.dates).dt.tz_localize(None)
        # we drop columns we do not need
        sum_tweet = sum_tweet[['week_start', 'scree_name', 'final_polarization', 'polarization_sum', 'domain_count']]
        return sum_tweet

    # Clean our dataframe
    df_clean = clean_df(df)
    sum_tweet = get_weekly_sum(df_clean)
    # Remove unnecessary columns
    df.drop(columns = ['domain_count', 'polarization_sum'], inplace = True)
    # we create a variable for weeks
    df['week_start'] = df['dates'].dt.to_period('W').dt.start_time
    # Then merge it with our weekly sum
    df = df.merge(sum_tweet, how = 'left', on = ['week_start', 'scree_name'])
    # we consider useful tweets for training only those ones of users who shared more than min_tweets URLs
    df['target_train'] = df.domain_shared.apply(lambda x: True if x >= min_tweets else False)


# 3. Merge Pre and Post Dataset -------------------------------------------------

# We first import our two main datasets

# POST COVID ----------------------
post_df = pd.read_csv(f"{path_to_raw}tweets_without_duplicates_regions_sentiment_demographics_topic_emotion.csv")

# PRE COVID ----------------------

# If it exist, then we have already our final dataset of pre-covid tweets
if os.path.exists(f"{path_to_data}processed/pred_final.pkl.gz"):
    print("Pre-covid dataset already merged")
    completed = pd.read_pickle(f"{path_to_data}processed/pred_final.pkl.gz", compression = 'gzip')
else:
    # We iteratively import all the batches of the topic predictions from the pre-covid dataset
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
    assert (len(failed)) == 0

    completed = pd.concat(completed)
    completed.to_pickle(f"{path_to_data}processed/pred_final.pkl.gz", compression = 'gzip')

    del tbd, failed, batch

    # # # Remove the single batches
    for i in range(0, size):
        if os.path.exists(f'{path_to_raw}batches/batch_{i}.pkl.gz'): os.remove(f'{path_to_raw}batches/batch_{i}.pkl.gz')

# 4. Create the training dataset -------------------------------------------------

# PARAMETERS
frequency = 'W-MON' # Set frequency of our dataset - we set it to weekly frequency, monday
min_tweets = 3 # Set minimum number of tweets

# Find all the URLs
find_urls(completed)
find_urls(post_df)

# Get the unshortened version
find_full_link(completed)
find_full_link(post_df)

# Apply it to our pre-covid dataset
extract_domains(completed)
extract_domains(post_df)

# Clean the domains
clean_domains(completed)
clean_domains(post_df)

print(f"Left domains: {left_doms_count}")
print(f"Right domains: {right_doms_count}")

# Get the domain count and polarization of each tweet
df_polarization(completed)
print(f"Count of valid domains:\n{completed.domain_count.value_counts()}")
df_polarization(post_df)
print(f"Count of valid domains:\n{post_df.domain_count.value_counts()}")

# Get the total polarization per tweet - when multiple links are present
tot_polarization(completed)
tot_polarization(post_df)

# Get the average weekly polarization
get_weekly_pol(completed)
get_weekly_pol(post_df)

# we limit our training set to tweets talking about politics
training_pre = completed.loc[(completed.target_train == True) & ((completed.final_pred == 1) | (completed.final_pred == 2))]
training_post = post_df.loc[(post_df.target_train == True) & ((post_df.topic == 'politics') | (post_df.topic == 'economics'))]

training = pd.concat([training_pre, training_post])
training.to_pickle(f"{path_to_data}processed/training.pkl.gz", compression = 'gzip')