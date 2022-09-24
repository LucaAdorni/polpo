###########################################################
# Code to unshorten URLs from our Twitter dataset
# Author: Luca Adorni
# Date: September 2022
###########################################################

# 0. Setup -------------------------------------------------

import numpy as np
import pandas as pd
import sys
import os
import re
import dill
import pickle
from tqdm import tqdm
from pytimedinput import timedInput
sys.path.append(f"{os.getcwd()}\\code\\utils")

from utils import Worker, ThreadPool, iterate_pool, clearConsole

# Setup Repository
with open("repo_info.txt", "r") as repo_info:
    path_to_repo = repo_info.readline()

path_to_data = f"{path_to_repo}data/"
path_to_raw = f"{path_to_data}raw/"
path_to_links = f"{path_to_data}links/"


# 1. Import URLs -------------------------------------------------

# Import all URLs we need to unshorten
with open(f'{path_to_links}url_list.pkl', 'rb') as f: 
    url_list = pickle.load(f)

# Import the dictionary of already unshortened URLs
try:
    with open(f'{path_to_links}url_dictionary.pkl', 'rb') as f: 
        url_proc = pickle.load(f)
    print("URL dictionary loaded")
except:
    url_proc = {}
    print("New URL dictionary initiated")

# Now remove all the URLs we have already unshortened
url_list = [url for url in url_list if url not in url_proc.keys()]
print(f"URLs to be unshortened: {len(url_list)}")

# 2. Unshorten URLs -------------------------------------------------

# PARAMETERS
minimum = 0 # beginning of our list
maximum = len(url_list) # end of our list len(new_urls)
steps = 10000 # how many urls per batch
range_list = list(np.arange(minimum,maximum,steps))

iter_count = 0
timeout_loop = 5

# signal.signal(signal.SIGINT, signal_handler)
timedOut = True
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
        batch_dict = iterate_pool(min_r, max_r, url_list) # our iterating thread pool to request urls
        url_proc = {**url_proc, **batch_dict} # we merge the results we already have with the ones of the current batch
        print("Batch ended")
        iter_count += steps
        if iter_count % 50000 == 0 or max_r == maximum:
            with open(f'{path_to_links}url_dictionary.pkl', 'wb') as f: pickle.dump(url_proc, f)
            print("Output Saved")
            timeout_loop = 30
    else:
        with open(f'{path_to_links}url_dictionary.pkl', 'wb') as f: pickle.dump(url_proc, f)
        print("Loop terminated and output saved")
        break

clearConsole()