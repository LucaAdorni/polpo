###########################################################
# TOPIC MODELING - Code to produce cleaned up text for LDA models
# Author: Luca Adorni
# Date: March 2023
###########################################################

# 0. Setup -------------------------------------------------

import re
import nltk
import gensim
from nltk.stem import WordNetLemmatizer
import spacy
from nltk.corpus import stopwords
import os
import sys
from gensim.models import Phrases
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from nltk.stem.snowball import SnowballStemmer
import warnings
import datetime as dt
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.simplefilter("ignore")



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
path_to_figure_odds = f"{path_to_figures}logit_res/"
path_to_tables = f"{path_to_repo}tables/"
path_to_ctm = f"{path_to_data}ctm/"
path_to_lda = f"{path_to_data}lda/"
path_to_gsdmm = f"{path_to_data}gsdmm/"

os.makedirs(path_to_ctm, exist_ok=True)
os.makedirs(path_to_lda, exist_ok=True)
os.makedirs(path_to_gsdmm, exist_ok=True)

# define a string of punctuation symbols
punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'
# Load italian stopwords
stop_words = stopwords.words('italian')
from numpy import loadtxt
stop2 = loadtxt(f"{path_to_data}stopwords.txt", dtype = str, comments="#", delimiter=",", unpack=False)
stop_words = stop_words + list(stop2)
stop_words = list(set(stop_words))

# Load SpaCy IT model
nlp = spacy.load("it_core_news_sm")

# we initialize our stemmer
stemmer = SnowballStemmer("italian", ignore_stopwords=True)

# PARAMETER ---------
stem = True
stem_tag = np.where(stem, "_stemm","")

# 1. Define main functions --------------------------------------------------------

# functions to clean tweets
def remove_links(tweet):
    """Takes a string and removes web links from it"""
    tweet = re.sub(r'http\S+', '', tweet)   # remove http links
    tweet = re.sub(r'bit.ly/\S+', '', tweet)  # remove bitly links
    tweet = tweet.strip('[link]')   # remove [links]
    tweet = re.sub(r'pic.twitter\S+','', tweet)
    return tweet


def remove_users(tweet):
    """Takes a string and removes retweet and @user information"""
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)  # remove re-tweet
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)  # remove tweeted at
    return tweet


def remove_hashtags(tweet):
    """Takes a string and removes any hash tags"""
    # tweet = re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)  # remove hash tags
    tweet = re.sub('#+', '', tweet)  # remove hash tags
    return tweet

def remove_keywords(tweet):
    """Takes a string and removes the main keywords"""
    tweet = re.sub('(coronavirus|covid19|covid-19|covid19italia|coronavid19|pandemia|corona|virus|covid)', '', tweet)  # remove hash tags
    # vaccino keyword is wrongly recognized by NLP SpaCy model, need to replace it
    tweet = re.sub('vaccino', "vaccinare", tweet)
    return tweet

def remove_av(tweet):
    """Takes a string and removes AUDIO/VIDEO tags or labels"""
    tweet = re.sub('VIDEO:', '', tweet)  # remove 'VIDEO:' from start of tweet
    tweet = re.sub('AUDIO:', '', tweet)  # remove 'AUDIO:' from start of tweet
    return tweet


def tokenize(tweet):
    """Returns tokenized representation of words in lemma form excluding stopwords"""
    result = []
    for token in gensim.utils.simple_preprocess(tweet):
        if token not in stop_words \
                and len(token) > 2:  # drops words with less than 3 characters
            if stem == True:
                result.append(stemmer.stem(token))
            else:
                result.append(nlp(token)[0].lemma_)
    return result


def preprocess_tweet(tweet):
    """Main master function to clean tweets, stripping noisy characters, and tokenizing use lemmatization"""
    tweet = remove_users(tweet)
    tweet = remove_links(tweet)
    tweet = remove_hashtags(tweet)
    tweet = remove_av(tweet)
    tweet = tweet.lower()  # lower case
    tweet = remove_keywords(tweet)
    tweet = re.sub('[' + punctuation + ']+', ' ', tweet)  # strip punctuation
    tweet = re.sub('\s+', ' ', tweet)  # remove double spacing
    tweet = re.sub('([0-9]+)', '', tweet)  # remove numbers
    tweet_token_list = tokenize(tweet)  # apply lemmatization and tokenization
    tweet = ' '.join(tweet_token_list)
    return tweet


def tokenize_tweets(df):
    """Main function to read in and return cleaned and preprocessed dataframe.
    This can be used in Jupyter notebooks by importing this module and calling the tokenize_tweets() function
    Args:
        df = data frame object to apply cleaning to
    Returns:
        pandas data frame with cleaned tokens
    """

    df['tokens'] = df.tweet_text.apply(preprocess_tweet)
    num_tweets = len(df)
    print('Complete. Number of Tweets that have been cleaned and tokenized : {}'.format(num_tweets))
    return df

def prepare_tokens(docs, min_bigrams = 20):
    bigram = Phrases(docs, min_count = min_bigrams)
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)
    return docs


# 2. Load our dataset --------------------------------------------

try:
    final_df = pd.read_pickle(f"{path_to_processed}tweet_for_topic.pkl.gz", compression = 'gzip')
    print("Loaded Data")
except:
    print("Creating data")
    # Load our predicted dataset
    df = pd.read_pickle(f"{path_to_processed}final_df_clean.pkl.gz", compression = 'gzip')
    # Keep only the info we want
    df = df[['week_start', 'scree_name', 'polarization_bin', 'prediction', 'extremism_toleft', 'extremism_toright', 'orient_change_toleft', 'orient_change_toright']]

    # Load back the tweets
    post_df = pd.read_csv(f"{path_to_raw}tweets_without_duplicates_regions_sentiment_demographics_topic_emotion.csv")
    post_df['dates'] = pd.to_datetime(post_df.dates)
    post_df['week_start'] = post_df['dates'].dt.to_period('W').dt.start_time
    # Drop columns we do not need
    post_df.drop(columns = ['tweet_ids', 'user_description' ,'locations', 'dates', 'place', 'polygon', 'user_id', 'profile_url', 'is_org'], inplace = True)

    # Restrict to 2020
    df = df.loc[(df.week_start <= pd.datetime(2020, 12, 31))& (df.week_start > pd.datetime(2019, 12, 31))]

    # Now merge the two sets back
    final_df = post_df.merge(df, how = 'inner', on = ['week_start', 'scree_name'], validate = 'm:1')

    # Save it
    final_df.to_pickle(f"{path_to_processed}tweet_for_topic.pkl.gz", compression = 'gzip')


# Restrict to topics we care about
final_df = final_df.loc[final_df.topic.isin(['politics', 'economics', 'health', 'vaccine'])]

# Tokenize our tweets
final_df = tokenize_tweets(final_df)
# Save it
final_df.to_pickle(f"{path_to_processed}cleaned_gsdmm_tweets{stem_tag}.pkl.gz", compression = 'gzip')