###########################################################
# Code to predict polarization
# Author: Luca Adorni
# Date: January 2023
###########################################################

# 0. Setup -------------------------------------------------

import numpy as np
import pandas as pd
import os
import sys
import re
import pickle
import random
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import dill
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import dill
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from catboost import CatBoostRegressor
import lightgbm
import xgboost as xgb

try:
    # Setup Repository
    with open("repo_info.txt", "r") as repo_info:
        path_to_repo = repo_info.readline()
except:
    path_to_repo = f"{os.getcwd()}/polpo/"
    sys.path.append(f"{os.getcwd()}/.local/bin") # import the temporary path where the server installed the module


print(path_to_repo)

path_to_data = f"{path_to_repo}data/"
path_to_figures = f"{path_to_repo}figures/"
path_to_tables = f"{path_to_repo}tables/"
path_to_raw = f"{path_to_data}raw/"
path_to_links = f"{path_to_data}links/"
path_to_topic = f"{path_to_data}topic/"
path_to_results = f"{path_to_data}results/"
path_to_models = f"{path_to_data}models/"
path_to_models_pol = f"{path_to_models}ML/"
path_to_processed = f"{path_to_data}processed/"


# PARAMETERS ---------------
random_seed = 42
random.seed(random_seed)

estimator = 'catboost'
method = 'frequency'
tune_models = True

if tune_models:
    tune_tag = '_tuned'
else:
    tune_tag = ''
# ----------------------

# 1. Import our dataset -------------------------------------------------

try:
    # Load our dataset, if we have already processed it
    x_pred = pd.read_pickle(f"{path_to_processed}df_processed.pkl.gz", compression = 'gzip')
    print("Dataset Loaded")
    print(x_pred.shape[0])
except:
    # PRE COVID ----------------------
    pre_df = pd.read_pickle(f"{path_to_data}processed/pred_final.pkl.gz", compression = 'gzip')

    # For the pre-covid data, change the topic column
    pre_df.loc[pre_df.final_pred == 1, 'topic'] = 'economics'
    pre_df.loc[pre_df.final_pred == 2, 'topic'] = 'politics'

    pre_df = pre_df.loc[(pre_df.topic == 'politics') | (pre_df.topic == 'economics')]

    # Get the beginning of weeks
    pre_df.dates = pd.to_datetime(pre_df.dates)
    pre_df['week_start'] = pre_df['dates'].dt.to_period('W').dt.start_time
    # And remove unnecessary columns
    pre_df = pre_df[['scree_name', 'tweet_text', 'week_start']]

    # POST COVID ----------------------
    post_df = pd.read_csv(f"{path_to_raw}tweets_without_duplicates_regions_sentiment_demographics_topic_emotion.csv")

    post_df = post_df.loc[post_df.is_org == 'non-org']

    # we limit our training set to tweets talking about politics
    post_df = post_df.loc[(post_df.topic == 'politics') | (post_df.topic == 'economics')]

    # Get the beginning of weeks
    post_df.dates = pd.to_datetime(post_df.dates)
    post_df['week_start'] = post_df['dates'].dt.to_period('W').dt.start_time
    # And remove unnecessary columns
    post_df = post_df[['scree_name', 'tweet_text', 'week_start']]

    # Merge the tweets
    post_df = pd.concat([post_df, pre_df])
    pred = post_df.groupby(['scree_name', 'week_start']).tweet_text.apply(' '.join).reset_index()
    del post_df, pre_df
    # Drop any duplicates
    pred.drop_duplicates(subset = ['scree_name', 'week_start'], inplace = True)

    pred.to_pickle(f"{path_to_processed}df_processed.pkl.gz", compression = 'gzip')

#     # Vectorize our datasets -------------------------------------------------

#     MAX_FEATURES = 10000 # maximum number of features
#     min_df = 5 # minimum frequency
#     max_df = 0.8 # maximum frequency
#     N_GRAM = (1,2) # n_gram range

#     STOPWORDS = stopwords.words("italian")
#     # we initialize our stemmer
#     stemmer = SnowballStemmer("italian", ignore_stopwords=True)

#     def text_prepare(text) :
#         """
#             text: a string        
#             return: modified initial string
#         """
            
#         text = text.lower() # lowercase text
#         text = emoji.demojize(text) # convert emojis to text
#         text = unidecode((text))
#         text = re.sub("#|@", "", text) # take away hashtags or mentions but keep the word
#         text = re.sub(r'(@[A-Za-z0–9_]+)|[^\w\s]|#|http\S+', "", text)
#         text =  " ".join([x for x in text.split()if x not in STOPWORDS]) # delete stopwords from text
#         text =  " ".join([stemmer.stem(x) for x in text.split()])
#         text =  " ".join([x for x in text.split()])
#         return text

#     print("Cleaning dataset")
#     pred["text_final"] = pred.tweet_text.apply(lambda x: text_prepare(x))

#     # Load back the training set
#     train = pd.read_pickle(f"{path_to_processed}train_clean.pkl.gz", compression = 'gzip')

#     def vectorize_to_dataframe(df, vectorizer_obj):
#         """
#         Function to return a dataframe from our vectorizer results
#         """
#         df = pd.DataFrame(data = df.toarray(), columns = vectorizer_obj.get_feature_names())
#         return df

#     def vectorize_features(X_train, X_test, method = 'frequency', include_val = False, X_val = ''):
#         """
#         Function to perform vectorization of our test sets
#         X_train, X_test, X_val: our dataframes
#         method: either 'frequency', 'tf_idf', 'onehot' to employ a different BoW technique
#         include_val: set to True if we also have a validation dataset
#         """
#         # initialize our vectorizer
#         if method == 'tf_idf':
#             vectorizer = TfidfVectorizer(ngram_range=N_GRAM, min_df=min_df, max_df=max_df, max_features=MAX_FEATURES)
#         elif method == 'frequency':
#             vectorizer = CountVectorizer(ngram_range=N_GRAM, min_df=min_df, max_df=max_df, max_features=MAX_FEATURES)
#         elif method == 'onehot':
#             vectorizer = CountVectorizer(ngram_range=N_GRAM, min_df=min_df, max_df=max_df, max_features=MAX_FEATURES, binary = True)
            
#         X_train = vectorizer.fit_transform(X_train.text_final)
#         X_train = vectorize_to_dataframe(X_train, vectorizer)
#         X_test = vectorizer.transform(X_test.text_final)
#         X_test = vectorize_to_dataframe(X_test, vectorizer)
#         if include_val: 
#             X_val = vectorizer.transform(X_val.text_final)
#             X_val = vectorize_to_dataframe(X_val, vectorizer)
#         return X_train, X_test, X_val

#     print("Count vectorizer")
#     X_train, x_pred, _ = vectorize_features(train, pred, method = method, include_val = False)

#     x_pred.to_pickle(f"{path_to_processed}df_processed.pkl.gz", compression = 'gzip')

# # 2. Load our best ML model -------------------------------------------------

# with open(f'{path_to_models_pol}/_{estimator}_{method}{tune_tag}', 'rb') as file:
#     model = dill.load(file)
# print('Model already trained')

# # Now predict over the whole dataset
# preds = model.predict(x_pred)

# with open(f'{path_to_processed}pred_pol.pkl', 'wb') as f: 
#     pickle.dump(preds, f)