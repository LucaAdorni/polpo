###########################################################
# Code to extract URLs from our Twitter dataset
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

# 1. Imports -------------------------------------------------

import numpy as np
import pandas as pd
import sys
import os
import re
import pickle
# Import libraries
from concurrent.futures import ThreadPoolExecutor
import requests
import time
import sys, signal

from urllib.request import Request, urlopen
#from fake_useragent import UserAgent
import random
from bs4 import BeautifulSoup
import requests
from urllib3.util.retry import Retry
from urllib3.exceptions import MaxRetryError, ProxySchemeUnknown, ProtocolError

try:
    # Setup Repository
    with open("repo_info.txt", "r") as repo_info:
        path_to_repo = repo_info.readline()
except:
    path_to_repo = f"{os.getcwd()}/polpo/"
    sys.path.append(f"{os.getcwd()}/.local/bin") # import the temporary path where the server installed the module

path_to_data = f"{path_to_repo}data/"
path_to_links = f"{path_to_data}links/"

# 2. Custom Functions -------------------------------------------------

results = {}
session = requests.session()
def unshorten_url(url):
    """
    Function to unshorten a single URL
    """
    try: r = session.head(url, timeout = 10).headers.get('location')
    except: r = 'Error'
    results[url] = r

def iterate_pool(min_r, max_r, new_urls, pool_n = 10000):
    """
    Function to initialize our pool url request in a for loop environment
    min_r: our initial link
    max_r: our final link
    new_urls: url list to be processed
    pool_n: number of Thread Pools we want to initialize
    """
    batch = new_urls[min_r:max_r]  # batch to be processed
    pool = ThreadPoolExecutor(pool_n)  # initialize a thread pool with 10k threads
    results = {}  # initialize our result dictionary
    session = requests.session()
    session.proxies.update({'http': f'http://{working_proxies[random_proxy(working_proxies)]}'})

    def unshorten_url(url):
        """
        Function to unshorten a single URL
        """
        try: r = session.head(url, timeout = 10).headers.get('location')
        except: r = 'Error'
        results[url] = r

    pool.map(unshorten_url, batch)  # process our batch
    pool.shutdown()
    return results

def check_proxy(proxy):
    """
    Function to properly check each proxy, taken from: https://stackoverflow.com/questions/72747987/trouble-selecting-functional-proxies-from-a-list-of-proxies-quickly
    """
    try:
        session = requests.Session()
        session.headers['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.131 Safari/537.36'
        session.max_redirects = 300
        proxy = proxy.split('\n', 1)[0]
        req = session.get("http://google.com", proxies={'http':'http://' + proxy}, timeout=10, allow_redirects=True)
        if req.status_code == 200:
            return proxy

    except requests.exceptions.ConnectTimeout as e:
        return None
    except requests.exceptions.ConnectionError as e:
        return None
    except ConnectionResetError as e:
        return None
    except requests.exceptions.HTTPError as e:
        return None
    except requests.exceptions.Timeout as e:
        return None
    except ProxySchemeUnknown as e:
        return None
    except ProtocolError as e:
        return None
    except requests.exceptions.ChunkedEncodingError as e:
        return None
    except requests.exceptions.TooManyRedirects as e:
        return None

# Retrieve a random index proxy (we need the index to delete it if not working)
def random_proxy(working_proxies):
  return random.randint(0, len(working_proxies) - 1)

def get_proxies():
    """
    Code partly taken from: https://stackoverflow.com/questions/38785877/spoofing-ip-address-when-web-scraping-python
    """
    # Here I provide some proxies for not getting caught while scraping
    #ua = UserAgent() # From here we generate a random user agent
    proxies = [] # Will contain proxies [ip, port]
    ua = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.131 Safari/537.36'
    # Retrieve latest proxies - for now just from ssl proxies
    proxies_req = Request('https://www.sslproxies.org/')
    proxies_req.add_header('User-Agent', ua)
    #proxies_req.add_header('User-Agent', ua.random)
    proxies_doc = urlopen(proxies_req).read().decode('utf8')
    # Now we parse the table through BeautifulSoup
    soup = BeautifulSoup(proxies_doc, 'html.parser')
    proxies_table = soup.find("table", {"class" : 'table table-striped table-bordered'})
    # Save proxies in the array
    for row in proxies_table.tbody.find_all('tr'):
        proxies.append(row.find_all('td')[0].string + ":" + row.find_all('td')[1].string)
    return proxies

def check_proxy_list(proxies):    
    # Iteratively check all the proxies we have, and keep only the functioning ones
    working_proxies = []
    for i in range(0,len(proxies)):
        working_proxies.append(check_proxy(proxies[i]))

    # Clean of the None values our list of working proxies
    working_proxies = [proxy for proxy in working_proxies if proxy != None]
    return working_proxies

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

# 2. STEP A: Download List of Proxies -------------------------------------------------

print("Process - Rank: %d -----------------------------------------------------"%comm.rank)

# Only one node will scrape the proxy list
if rank == 0:
    proxies = get_proxies()
    if mpi_use:
        proxies = proxies
        proxies = split_list(proxies)
else:
    proxies = None

# Now we send our splitted list of proxies to our nodes
if mpi_use:
    proxies = comm.scatter(proxies, root = 0)

working_proxies = check_proxy_list(proxies)

if mpi_use:
    working_proxies = comm.gather(working_proxies, root = 0)

if rank == 0 and mpi_use == True:
    working_proxies = [link for sub_list in working_proxies for link in sub_list]
    # now we broadcast the final list to everyone

if mpi_use:
    working_proxies = comm.bcast(working_proxies, root=0)

print(f'Received Working list: {len(working_proxies)}')

# 3. STEP B: Load URL List -------------------------------------------------

# Only the first machine does that
if rank == 0:
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
        # Import all URLs we need to unshorten
        with open(f'{path_to_links}url_list.pkl', 'rb') as f: 
            url_list = pickle.load(f)
            print(f"URLs in dataset: {len(url_list)}")
        # Check what links we are missing
        url_list = [url for url in url_list if url not in url_dict.keys()]
        url_list = split_list(url_list)
        for i in range(0,size):
            with open(f'{path_to_links}url_list_{i}.pkl', 'wb') as f: 
                pickle.dump(url_list[i], f)
        del url_list, url_dict


# Import all URLs we need to unshorten
with open(f'{path_to_links}url_list_{rank}.pkl', 'rb') as f: 
    url_list = pickle.load(f)
    print(f"URLs in dataset: {len(url_list)}")

# PARAMETERS
minimum = 0 # beginning of our list
maximum = len(url_list) # end of our list len(new_urls)
steps = 100 # how many urls per batch
range_list = list(np.arange(minimum,maximum,steps))

iter_count = 0

url_proc = {}
for i in range(len(range_list)):
    print(i)
    min_r = range_list[i]
    try: 
        max_r = range_list[i+1]
    except: 
        max_r = maximum # if we are out of range we use the list up to its maximum
    batch_dict = iterate_pool(min_r, max_r, url_list, pool_n = steps) # our iterating thread pool to request urls
    url_proc = {**url_proc, **batch_dict} # we merge the results we already have with the ones of the current batch
    iter_count += steps
    if iter_count % 2000 == 0 or max_r == maximum:
        with open(f'{path_to_links}url_dictionary_{rank}.pkl', 'wb') as f: 
            pickle.dump(url_proc, f)
            print("saved dictionary")