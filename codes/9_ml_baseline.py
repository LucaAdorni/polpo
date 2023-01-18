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
from tabulate import tabulate
import dill
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from catboost import CatBoostRegressor
import lightgbm

##for clustering

from sklearn.linear_model import Lasso

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
path_to_alberto = "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0"

os.makedirs(path_to_figures, exist_ok = True)
os.makedirs(path_to_tables, exist_ok = True)
os.makedirs(path_to_topic, exist_ok = True)
os.makedirs(path_to_results, exist_ok = True)
os.makedirs(path_to_models, exist_ok = True)
os.makedirs(path_to_models_pol, exist_ok = True)


# 1. Parameters --------------------------------------------------------------------------------------------------

method_list = ['frequency', 'onehot','tf_idf']
random_seed = 42
random.seed(random_seed)
tune_models = True

if tune_models:
  tune_tag = '_tuned'
else:
  tune_tag = ''

# 2. Models ---------------------------------------------------------------------------------------------------

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


# Get latex code for the results
with open(f'{path_to_results}test_results{tune_tag}.pickle', 'rb') as handle:
    results_test = pickle.load(handle)


model_dict = {
    'rand_for': RandomForestRegressor(random_state = random_seed, n_jobs = -1)
    #,'lightgbm': lightgbm.LGBMRegressor(random_state = random_seed, n_jobs = -1)
    , 'lasso': Lasso(random_state = random_seed)
    , 'catboost': CatBoostRegressor(random_state = random_seed)
}

method_list = ['frequency', 'onehot','tf_idf']
def get_results(results_dict):
    mae = []
    mse = []
    r2 = []
    for estimator in model_dict.keys():
        mae_results = [estimator]
        mse_results = [estimator]
        r2_results = [estimator]
        for method in method_list:
            mae_results.append(results_dict[method][estimator]["MAE"])
            mse_results.append(results_dict[method][estimator]["MSE"])
            r2_results.append(results_dict[method][estimator]["R2"])
        mae.append(mae_results)
        mse.append(mse_results)
        r2.append(r2_results)
    return mae, mse, r2

mae_test, mse_test, r2_test = get_results(results_test)

print(tabulate(mae_test))


def get_results(results_dict):
    final_results = []
    for estimator in model_dict.keys():
        for method in method_list:
            final_results.append([estimator, method, "No", results_dict[method][estimator]["MAE"], results_dict[method][estimator]["MSE"], results_dict[method][estimator]["R2"]])
    return final_results

final_res = get_results(results_test)


table = tabulate(final_res, tablefmt = 'latex', headers = ['Model', 'Method', 'Tuning', 'MAE', "MSE", "R2"], floatfmt=".4f")

with open(f'{path_to_tables}table_3_ml_baseline.tex', 'w') as f:
    f.write(table)