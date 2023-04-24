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
import pickle
import sys
import tldextract
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import random
import re
from unidecode import unidecode
import emoji
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

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

# Option to remove the Center class from our labels
remove_cent = False

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

with open(f'{path_to_links}url_dictionary.pkl', 'rb') as f: 
    url_dict = pickle.load(f)

# 2. Define the functions to compute polarization -------------------------------------------------


# find all urls within the tweets, gives back a list
def find_urls(df):
    """
    Function to iteratively find all URLs within tweets
    """
    df["url"] = df.tweet_text.apply(lambda x: re.findall("(?P<url>https?://[^\s]+)", x))

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

    df.apply(lambda x: iterate_full_link(x, url_dict), axis = 1)
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

    df.apply(lambda x: iterate_extract(x), axis = 1)
    # Drop the full-url column since we do not need it anymore
    df.drop(columns = 'full_url', inplace = True)

# Load the domain alignments -----------------------------------------------------------

link_labels = pd.read_csv(f'{path_to_data}domain_alignment/italian_labels.csv')
if remove_cent:
    link_labels = link_labels.loc[link_labels.polarity_cont != 0]
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

    df['domain_clean']=df['domain'].apply(lambda x: clean_lr(x))
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

    df.apply(lambda x: iterate_polarization(x), axis = 1)
    
    # count the number of clean domains per observation
    df["domain_count"] = df.polarization.apply(lambda x: len(x))


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

    df['polarization_sum'] = df.polarization.apply(lambda x: polarization_sum(x))
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
    df['target_train'] = df.domain_count.apply(lambda x: True if x >= min_tweets else False)
    return df


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

# For the pre-covid data, change the topic column
completed.loc[completed.final_pred == 1, 'topic'] = 'economics'
completed.loc[completed.final_pred == 2, 'topic'] = 'politics'

# Get some basic information on the completed pre-training dataset
print(f"Pre-covid dataset size: {completed.shape[0]}")
print(f"Pre-covid number of users: {completed.scree_name.nunique()}")
print(f"Pre-covid topic distribution:\n{completed.topic.value_counts()}")
# Pre-covid dataset size: 36136560
# Pre-covid number of users: 385415
# Pre-covid topic distribution:
# 100          33370146
# politics      2127199
# economics      639215
# Name: topic, dtype: int64

print(f"Number of links unshortened: {len(url_dict)}")
# Number of links unshortened: 35477216

# Remove organizations (for the pre-covid we have already removed them)
post_df = post_df.loc[post_df.is_org == 'non-org']

training = pd.concat([post_df, completed])

# PARAMETERS
frequency = 'W-MON' # Set frequency of our dataset - we set it to weekly frequency, monday
min_tweets = 3 # Set minimum number of tweets

# Find all the URLs
find_urls(training)

# Get the unshortened version
find_full_link(training)

# Apply it to our pre-covid dataset
extract_domains(training)

# Clean the domains
clean_domains(training)

print(f"Left domains: {left_doms_count}")
print(f"Right domains: {right_doms_count}")

# Get the domain count and polarization of each tweet
df_polarization(training)
print(f"Count of valid domains:\n{training.domain_count.value_counts()}")

# Get the total polarization per tweet - when multiple links are present
tot_polarization(training)

# Get the average weekly polarization
training = get_weekly_pol(training)

# we limit our training set to tweets talking about politics
training = training.loc[(training.target_train == True) & ((training.topic == 'politics') | (training.topic == 'economics'))]

training.to_pickle(f"{path_to_data}processed/training.pkl.gz", compression = 'gzip')

# 5. Clean training dataset -------------------------------------------------------------

training = pd.read_pickle(f"{path_to_data}processed/training.pkl.gz", compression = 'gzip')
print(f"Training dataset size: {training.shape[0]}")
print(f"Training number of users: {training.scree_name.nunique()}")
print(f"Training topic distribution:\n{training.topic.value_counts()}")
# Training dataset size: 400326
# Training number of users: 8607
# Training topic distribution:
# politics     266652
# economics    133674
# Name: topic, dtype: int64


# Drop columns we do not need
training.drop(columns = ['topic_0', 'topic_1', 'topic_2', 'final_pred', 'domain_clean', 'user_description', 'place', 'polygon', 'profile_url', 'is_org', 'sentiment', 'emotion', 'gender', 'age', 'regions', 'locations'], inplace = True)

# Get info on the Users
post_df = pd.read_csv(f"{path_to_raw}tweets_without_duplicates_regions_sentiment_demographics_topic_emotion.csv")
post_df = post_df[['scree_name', 'regions', 'age', 'gender']]
post_df.drop_duplicates(subset = 'scree_name', inplace = True)

# Merge it with training dataset
training = training.merge(post_df, on = 'scree_name', how = 'left', indicator = True, validate = 'm:1')
assert training._merge.value_counts()["both"] == training.shape[0]
training.drop(columns = '_merge', inplace = True)

# Binarize our target
training['polarization_bin'] = pd.cut(x=training['final_polarization'], bins = [-1.01, -0.6, -0.2, 0.2, 0.6, 1.01], labels = ["far_left", "center_left", "center","center_right", "far_right"])
print(f"Distribution of target variable:\n{training.polarization_bin.value_counts()}")
# Distribution of target variable:
# center_left     147444
# center          140305
# center_right     82835
# far_left          3507
# far_right         1266
# Name: polarization_bin, dtype: int64

# Save our dataset
training.to_pickle(f"{path_to_data}processed/training.pkl.gz", compression = 'gzip')

# 6. Create Train/Test split -------------------------------------------------------------

training = pd.read_pickle(f"{path_to_data}processed/training.pkl.gz", compression = 'gzip')

# PARAMETERS
test_size = 0.1
val_size = 0.1
train_size = 1 - test_size - val_size
tag_size = '_02'
random_seed = 42
random.seed(random_seed)

merge_tweets = True # set to True if we want to merge all tweets
only_politics = False # set to True if we want to keep only politic tweets

if only_politics:
    politics_tag = '_politics'
else:
    politics_tag = ''

if merge_tweets:
    merged_tag = '_merged'
else:
    merged_tag = ''

# If only politics, restrict to only politic tweets
if only_politics:
    print(training.shape)
    training = training.loc[training.topic == 'politics']
    print(training.shape)

# If we want to merge all tweets per week/user
if merge_tweets:
    # Groupby join
    merged_tweets = training.groupby(['scree_name', 'week_start']).tweet_text.apply(' '.join).reset_index()
    # Get from the original dataset the final polarization score
    clean_pol = training[['scree_name', 'week_start', 'final_polarization', 'polarization_bin']].drop_duplicates()
    assert clean_pol.shape[0] == merged_tweets.shape[0]
    merged_tweets = merged_tweets.merge(clean_pol, on = ['scree_name', 'week_start'], validate = '1:1')
    # Drop any duplicates
    merged_tweets.drop_duplicates(subset = ['scree_name', 'week_start'], inplace = True)
else:
    merged_tweets = training.copy()

from sklearn.model_selection import StratifiedGroupKFold

cv = StratifiedGroupKFold(n_splits = 2, shuffle = True, random_state = random_seed)

# we first perform a first split between training set and validation sets
train_idxs, test_idxs = next(cv.split(merged_tweets, merged_tweets.week_start, groups = merged_tweets.scree_name))
df_train = merged_tweets.iloc[train_idxs]
df_test = merged_tweets.iloc[test_idxs]
# then we split from the test set our validation and test set
test_idxs, val_idxs = next(cv.split(df_test, df_test.week_start, groups = df_test.scree_name))
df_test_2 = df_test.iloc[test_idxs]
df_val = df_test.iloc[val_idxs]
df_test = df_test_2.copy()
# finally from our test set, to have a bigger training set, we split it again
val_idxs, train_idxs = next(cv.split(df_val, df_val.week_start, groups = df_val.scree_name))
df_train_2 = df_val.iloc[train_idxs]
df_val_2 = df_val.iloc[val_idxs]
df_val = df_val_2.copy()
df_train = df_train.append(df_train_2)

print(f'Training Set: {df_train.shape}')
print(f'Validation Set: {df_val.shape}')
print(f'Test Set: {df_test.shape}')
# Training Set: (21295, 5)
# Validation Set: (4160, 5)
# Test Set: (9156, 5)

print(f'Training Set: {df_train.scree_name.nunique()}')
print(f'Validation Set: {df_val.scree_name.nunique()}')
print(f'Test Set: {df_test.scree_name.nunique()}')
# Training Set: 5367
# Validation Set: 1067
# Test Set: 2173

print(f'Original # Weeks: {merged_tweets.week_start.nunique()}')
print(f'Training Set # Weeks: {df_train.week_start.nunique()}')
print(f'Validation Set # Weeks: {df_val.week_start.nunique()}')
print(f'Test Set # Weeks: {df_test.week_start.nunique()}')
# Training Set # Weeks: 98
# Validation Set # Weeks: 97
# Test Set # Weeks: 98

df_train.to_pickle(f"{path_to_data}processed/df_train{merged_tag}.pkl.gz", compression = 'gzip')
df_val.to_pickle(f"{path_to_data}processed/df_val{merged_tag}.pkl.gz", compression = 'gzip')
df_test.to_pickle(f"{path_to_data}processed/df_test{merged_tag}.pkl.gz", compression = 'gzip')

# 7. Machine Learning Files --------------------------------------------------------------------------------------------------

# For the ML Baseline, we start with the merged Train/Test/Val and then compute the various BoW methods
train = pd.read_pickle(f"{path_to_processed}df_train_merged.pkl.gz", compression = 'gzip')
val = pd.read_pickle(f"{path_to_processed}df_val_merged.pkl.gz", compression = 'gzip')
test = pd.read_pickle(f"{path_to_processed}df_test_merged.pkl.gz", compression = 'gzip')

# Transform our topics into labels
replace_dict = {"far_left": 0, "center_left": 1, "center":2, "center_right":3, "far_right":4}
train.polarization_bin.replace(replace_dict, inplace = True)
val.polarization_bin.replace(replace_dict, inplace = True)
test.polarization_bin.replace(replace_dict, inplace = True)
print(f"\nNumber of classes: {train.polarization_bin.nunique()}")


MAX_FEATURES = 10000 # maximum number of features
min_df = 5 # minimum frequency
max_df = 0.8 # maximum frequency
N_GRAM = (1,2) # n_gram range

STOPWORDS = stopwords.words("italian")
# we initialize our stemmer
stemmer = SnowballStemmer("italian", ignore_stopwords=True)

def text_prepare(text) :
    """
        text: a string        
        return: modified initial string
    """
        
    text = text.lower() # lowercase text
    text = emoji.demojize(text) # convert emojis to text
    text = unidecode((text))
    text = re.sub("#|@", "", text) # take away hashtags or mentions but keep the word
    text = re.sub(r'(@[A-Za-z0–9_]+)|[^\w\s]|#|http\S+', "", text)
    text =  " ".join([x for x in text.split()if x not in STOPWORDS]) # delete stopwords from text
    text =  " ".join([stemmer.stem(x) for x in text.split()])
    text =  " ".join([x for x in text.split()])
    return text

for df in [train, val, test]:  
    df["text_final"] = df.tweet_text.apply(lambda x: text_prepare(x))

# Save the dataframes
train.to_pickle(f"{path_to_processed}train_clean.pkl.gz", compression = 'gzip')
test.to_pickle(f"{path_to_processed}test_clean.pkl.gz", compression = 'gzip')
val.to_pickle(f"{path_to_processed}val_clean.pkl.gz", compression = 'gzip')

def vectorize_to_dataframe(df, vectorizer_obj):
    """
    Function to return a dataframe from our vectorizer results
    """
    df = pd.DataFrame(data = df.toarray(), columns = vectorizer_obj.get_feature_names())
    return df

def vectorize_features(X_train, X_test, method = 'frequency', include_val = False, X_val = ''):
    """
    Function to perform vectorization of our test sets
    X_train, X_test, X_val: our dataframes
    method: either 'frequency', 'tf_idf', 'onehot' to employ a different BoW technique
    include_val: set to True if we also have a validation dataset
    """
    # initialize our vectorizer
    if method == 'tf_idf':
        vectorizer = TfidfVectorizer(ngram_range=N_GRAM, min_df=min_df, max_df=max_df, max_features=MAX_FEATURES)
    elif method == 'frequency':
        vectorizer = CountVectorizer(ngram_range=N_GRAM, min_df=min_df, max_df=max_df, max_features=MAX_FEATURES)
    elif method == 'onehot':
        vectorizer = CountVectorizer(ngram_range=N_GRAM, min_df=min_df, max_df=max_df, max_features=MAX_FEATURES, binary = True)
        
    X_train = vectorizer.fit_transform(X_train.text_final)
    X_train = vectorize_to_dataframe(X_train, vectorizer)
    X_test = vectorizer.transform(X_test.text_final)
    X_test = vectorize_to_dataframe(X_test, vectorizer)
    if include_val: 
        X_val = vectorizer.transform(X_val.text_final)
        X_val = vectorize_to_dataframe(X_val, vectorizer)
    return X_train, X_test, X_val

method_list = ['frequency', 'onehot','tf_idf']

for method in method_list:
    X_train, X_test, X_val = vectorize_features(train, test, method = method, include_val = True, X_val = val)

    # Save the dataframes
    X_train.to_pickle(f"{path_to_processed}train_clean{method}.pkl.gz", compression = 'gzip')
    X_test.to_pickle(f"{path_to_processed}test_clean{method}.pkl.gz", compression = 'gzip')
    X_val.to_pickle(f"{path_to_processed}val_clean{method}.pkl.gz", compression = 'gzip')

# Save the dataframes
y_train = train[['final_polarization', 'polarization_bin']]
y_train.to_pickle(f"{path_to_processed}y_train.pkl.gz", compression = 'gzip')
y_test = test[['final_polarization', 'polarization_bin']]
y_test.to_pickle(f"{path_to_processed}y_test.pkl.gz", compression = 'gzip')
y_val = val[['final_polarization', 'polarization_bin']]
y_val.to_pickle(f"{path_to_processed}y_val.pkl.gz", compression = 'gzip')