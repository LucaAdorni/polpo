###########################################################
# Code to train a BERT model for polarization prediction
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
import multiprocessing
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from tabulate import tabulate
import dill
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from catboost import CatBoostRegressor
import lightgbm

##for clustering
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from sklearn.linear_model import Lasso
import re
from unidecode import unidecode
import emoji

try:
    # Setup Repository
    with open("repo_info.txt", "r") as repo_info:
        path_to_repo = repo_info.readline()
except:
    path_to_repo = f"{os.getcwd()}/polpo/"
    sys.path.append(f"{os.getcwd()}/.local/bin") # import the temporary path where the server installed the module


# PARAMETERS------------------
classification = True
  
print(path_to_repo)

path_to_data = f"{path_to_repo}data/"
path_to_figures = f"{path_to_repo}figures/"
path_to_raw = f"{path_to_data}raw/"
path_to_links = f"{path_to_data}links/"
path_to_topic = f"{path_to_data}topic/"
path_to_results = f"{path_to_data}results/"
path_to_models = f"{path_to_data}models/"
if classification:
    path_to_models_pol = f"{path_to_models}pol_class/"
else:
    path_to_models_pol = f"{path_to_models}pol_reg/"
path_to_processed = f"{path_to_data}processed/"
path_to_alberto = "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0"

os.makedirs(path_to_figures, exist_ok = True)
os.makedirs(path_to_topic, exist_ok = True)
os.makedirs(path_to_results, exist_ok = True)
os.makedirs(path_to_models, exist_ok = True)
os.makedirs(path_to_models_pol, exist_ok = True)

# 2. Read Files --------------------------------------------------------------------------------------------------

train = pd.read_pickle(f"{path_to_processed}df_train.pkl.gz", compression = 'gzip')
val = pd.read_pickle(f"{path_to_processed}df_val.pkl.gz", compression = 'gzip')
test = pd.read_pickle(f"{path_to_processed}df_test.pkl.gz", compression = 'gzip')

# Transform our topics into labels
replace_dict = {"far_left": 0, "center_left": 1, "center":2, "center_right":3, "far_right":4}
train.polarization_bin.replace(replace_dict, inplace = True)
val.polarization_bin.replace(replace_dict, inplace = True)
test.polarization_bin.replace(replace_dict, inplace = True)
print(f"\nNumber of classes: {train.polarization_bin.nunique()}")


# 3. Clean Datasets --------------------------------------------------------------------------------------------------


MAX_FEATURES = 10000 # maximum number of features
min_df = 5 # minimum frequency
max_df = 0.8 # maximum frequency
N_GRAM = (1,2) # n_gram range
random_seed = 42
random.seed(random_seed)
tune_models = True

if tune_models:
  tune_tag = '_tuned'
else:
  tune_tag = ''



# STOPWORDS = stopwords.words("italian")
# # we initialize our stemmer
# stemmer = SnowballStemmer("italian", ignore_stopwords=True)

# def text_prepare(text) :
#     """
#         text: a string        
#         return: modified initial string
#     """
        
#     text = text.lower() # lowercase text
#     text = emoji.demojize(text) # convert emojis to text
#     text = unidecode((text))
#     text = re.sub("#|@", "", text) # take away hashtags or mentions but keep the word
#     text = re.sub(r'(@[A-Za-z0–9_]+)|[^\w\s]|#|http\S+', "", text)
#     text =  " ".join([x for x in text.split()if x not in STOPWORDS]) # delete stopwords from text
#     text =  " ".join([stemmer.stem(x) for x in text.split()])
#     text =  " ".join([x for x in text.split()])
#     return text

# for df in [train, val, test]:  
#     df["text_final"] = df.tweet_text.apply(lambda x: text_prepare(x))

# # Save the dataframes
# train.to_pickle(f"{path_to_processed}train_clean.pkl.gz", compression = 'gzip')
# test.to_pickle(f"{path_to_processed}test_clean.pkl.gz", compression = 'gzip')
# val.to_pickle(f"{path_to_processed}val_clean.pkl.gz", compression = 'gzip')

# def vectorize_to_dataframe(df, vectorizer_obj):
#     """
#     Function to return a dataframe from our vectorizer results
#     """
#     df = pd.DataFrame(data = df.toarray(), columns = vectorizer_obj.get_feature_names())
#     return df

# def vectorize_features(X_train, X_test, method = 'frequency', include_val = False, X_val = ''):
#     """
#     Function to perform vectorization of our test sets
#     X_train, X_test, X_val: our dataframes
#     method: either 'frequency', 'tf_idf', 'onehot' to employ a different BoW technique
#     include_val: set to True if we also have a validation dataset
#     """
#     # initialize our vectorizer
#     if method == 'tf_idf':
#         vectorizer = TfidfVectorizer(ngram_range=N_GRAM, min_df=min_df, max_df=max_df, max_features=MAX_FEATURES)
#     elif method == 'frequency':
#         vectorizer = CountVectorizer(ngram_range=N_GRAM, min_df=min_df, max_df=max_df, max_features=MAX_FEATURES)
#     elif method == 'onehot':
#         vectorizer = CountVectorizer(ngram_range=N_GRAM, min_df=min_df, max_df=max_df, max_features=MAX_FEATURES, binary = True)
        
#     X_train = vectorizer.fit_transform(X_train.text_final)
#     X_train = vectorize_to_dataframe(X_train, vectorizer)
#     X_test = vectorizer.transform(X_test.text_final)
#     X_test = vectorize_to_dataframe(X_test, vectorizer)
#     if include_val: 
#         X_val = vectorizer.transform(X_val.text_final)
#         X_val = vectorize_to_dataframe(X_val, vectorizer)
#     return X_train, X_test, X_val

method_list = ['frequency', 'onehot','tf_idf']

# for method in method_list:
#     X_train, X_test, X_val = vectorize_features(train, test, method = method, include_val = True, X_val = val)

#     # Save the dataframes
#     X_train.to_pickle(f"{path_to_processed}train_clean{method}.pkl.gz", compression = 'gzip')
#     X_test.to_pickle(f"{path_to_processed}test_clean{method}.pkl.gz", compression = 'gzip')
#     X_val.to_pickle(f"{path_to_processed}val_clean{method}.pkl.gz", compression = 'gzip')

# # Save the dataframes
# y_train = train[['final_polarization', 'polarization_bin']]
# y_train.to_pickle(f"{path_to_processed}y_train.pkl.gz", compression = 'gzip')
# y_test = test[['final_polarization', 'polarization_bin']]
# y_test.to_pickle(f"{path_to_processed}y_test.pkl.gz", compression = 'gzip')
# y_val = val[['final_polarization', 'polarization_bin']]
# y_val.to_pickle(f"{path_to_processed}y_val.pkl.gz", compression = 'gzip')


# 4. Models ---------------------------------------------------------------------------------------------------

def evaluate_model(X_df, y_df, model):
    preds = model.predict(X_df)
    mae = mean_absolute_error(y_df, preds)
    mse = mean_squared_error(y_df, preds)
    r_squared = r2_score(y_df, preds)
    print(f'MAE: {mae: .3f}')
    print(f'MSE: {mse: .3f}')
    print(f'R2: {r_squared: .3f}')
    final_results = {'MAE': mae, 'MSE': mse, 'R2': r_squared}
    return final_results


model_dict = {
    'rand_for': RandomForestRegressor(random_state = random_seed, n_jobs = -1)
    #,'lightgbm': lightgbm.LGBMRegressor(random_state = random_seed, n_jobs = -1)
    , 'lasso': Lasso(random_state = random_seed)
    , 'catboost': CatBoostRegressor(random_state = random_seed)
}

# PARAMETERS FOR LOGISTIC REGRESSION -------
param_en = {'alpha': list(np.arange(0,1.1,0.1))}

# PARAMETERS FOR DECISION TREE -------------
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# PARAMETERS FOR RANDOM FOREST -------
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 5)]

# Maximum number of samples per tree
max_sampl = list(np.arange(0.01,1,0.2))
max_sampl.append(None)
# Create the random grid
param_rf = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'max_samples': max_sampl}

# PARAMETERS FOR GRADIENT BOOSTING --------

learn_rate = list(np.linspace(0.01, 1, num = 10))


# PARAMETERS FOR LIGHTGBM -----------
param_lgb = {'max_depth': max_depth,
             'min_data_in_leaf': min_samples_leaf,
             'num_iterations': n_estimators,
             'learning_rate': learn_rate,
             'colsample_bytree': list(np.linspace(0.1, 1, num = 10)),
             'subsample': list(np.linspace(0.1, 1, num = 10)),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

# PARAMETERS FOR CATBOOSTING ------

param_cat = {'iterations': n_estimators,
            'learning_rate': learn_rate,
               'rsm': list(np.linspace(0.1, 1, num = 10)),
               'depth': [int(x) for x in np.linspace(2, 16, num = 11)]
            , 'l2_leaf_reg': [1, 2, 3, 4, 5, 7, 9, 15, 20]}



param_dictionary = {
    'rand_for': param_rf
    ,'lightgbm': param_lgb
    , 'lasso': param_en
    , 'catboost': param_cat
}


results_train = {}
results_val = {}
results_test = {}

y_train = pd.read_pickle(f"{path_to_processed}y_train.pkl.gz", compression = 'gzip')
y_test = pd.read_pickle(f"{path_to_processed}y_test.pkl.gz", compression = 'gzip')
y_val = pd.read_pickle(f"{path_to_processed}y_val.pkl.gz", compression = 'gzip')


for method in method_list:
    print(method)
    # Load the dataframes
    X_train = pd.read_pickle(f"{path_to_processed}train_clean{method}.pkl.gz", compression = 'gzip')
    X_test = pd.read_pickle(f"{path_to_processed}test_clean{method}.pkl.gz", compression = 'gzip')
    X_val = pd.read_pickle(f"{path_to_processed}val_clean{method}.pkl.gz", compression = 'gzip')
    print("Dataframes successfully loaded")
    train_res = {}
    val_res = {}
    test_res = {}
    for estimator in model_dict.keys():
        print(estimator)
        try:
            with open(f'{path_to_models_pol}/_{estimator}_{method}{tune_tag}', 'rb') as file:
                model = dill.load(file)
            print('Model already trained')
        except:
            print('Fitting Model')
            model = model_dict[estimator]
            if tune_models:
                gridsearch = RandomizedSearchCV(model, param_dictionary[estimator], cv = 5, n_jobs = -1)
                gridsearch.fit(X_train, y_train.final_polarization)
                model = gridsearch.best_estimator_
            else:
                model.fit(X_train, y_train.final_polarization)

            with open(f'{path_to_models_pol}/_{estimator}_{method}{tune_tag}', 'wb') as file:
                dill.dump(model, file)
            print('Model saved')    
        train_res[estimator] = evaluate_model(X_train, y_train.final_polarization, model)
        val_res[estimator] = evaluate_model(X_val, y_val.final_polarization, model)
        test_res[estimator] = evaluate_model(X_test, y_test.final_polarization, model)
        results_train[method] = train_res
        results_val[method] = val_res
        results_test[method] = test_res



with open(f'{path_to_results}train_results{tune_tag}.pickle', 'wb') as handle:
    pickle.dump(results_train, handle)
with open(f'{path_to_results}val_results{tune_tag}.pickle', 'wb') as handle:
    pickle.dump(results_val, handle)
with open(f'{path_to_results}test_results{tune_tag}.pickle', 'wb') as handle:
    pickle.dump(results_test, handle)