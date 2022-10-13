# Import libraries
import numpy as np
import pandas as pd
import os
import re
import dill
from tqdm import tqdm as tqdm
import time
import pickle
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import requests
from threading import Thread
import time
import sys, signal

from urllib.request import Request, urlopen
from fake_useragent import UserAgent
import random
from bs4 import BeautifulSoup
from IPython.core.display import clear_output
import requests
from urllib3.util.retry import Retry
from urllib3.exceptions import MaxRetryError, ProxySchemeUnknown, ProtocolError

# UNSHORTEN URLS ---------------------------------------------------------------

class Worker(Thread):
    """ Thread executing tasks from a given tasks queue """
    def __init__(self, tasks):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.start()

    def run(self):
        while True:
            func, args, kargs = self.tasks.get()
            try:
                func(*args, **kargs)
            except Exception as e:
                # An exception happened in this thread
                print(e)
            finally:
                # Mark this task as done, whether an exception happened or not
                self.tasks.task_done()


class ThreadPool:
    """ Pool of threads consuming tasks from a queue """

    def __init__(self, num_threads):
        self.tasks = Queue(num_threads)
        for _ in range(num_threads):
            Worker(self.tasks)

    def add_task(self, func, *args, **kargs):
        """ Add a task to the queue """
        self.tasks.put((func, args, kargs))

    def map(self, func, args_list):
        """ Add a list of tasks to the queue """
        for args in args_list:
            self.add_task(func, args)

    def wait_completion(self):
        """ Wait for completion of all the tasks in the queue """
        self.tasks.join()


def iterate_pool(min_r, max_r, new_urls, pool_n = 10000):
    """
    Function to initialize our pool url request in a for loop environment
    min_r: our initial link
    max_r: our final link
    new_urls: url list to be processed
    pool_n: number of Thread Pools we want to initialize
    """
    batch = new_urls[min_r:max_r]  # batch to be processed
    pool = ThreadPool(pool_n)  # initialize a thread pool with 10k threads
    results = {}  # initialize our result dictionary
    session = requests.session()
    session.proxies.update({'http': f'http://{working_proxies[random_proxy(working_proxies)]}'})

    now = time.time()

    def unshorten_url(url):
        """
        Function to unshorten a single URL
        """
        try: r = session.head(url, timeout = 10).headers.get('location')
        except: r = 'Error'
        results[url] = r

    pool.map(unshorten_url, batch)  # process our batch
    pool.wait_completion()
    time_taken = time.time() - now
    print("Time taken: {}".format(time_taken))  # print total time
    return results

def clearConsole():
    command = 'clear'
    if os.name in ('nt', 'dos'):  # If Machine is running on Windows, use cls
        command = 'cls'
    os.system(command)

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
    ua = UserAgent() # From here we generate a random user agent
    proxies = [] # Will contain proxies [ip, port]

    # Retrieve latest proxies - for now just from ssl proxies
    proxies_req = Request('https://www.sslproxies.org/')
    proxies_req.add_header('User-Agent', ua.random)
    proxies_doc = urlopen(proxies_req).read().decode('utf8')
    # Now we parse the table through BeautifulSoup
    soup = BeautifulSoup(proxies_doc, 'html.parser')
    proxies_table = soup.find("table", {"class" : 'table table-striped table-bordered'})
    # Save proxies in the array
    for row in proxies_table.tbody.find_all('tr'):
        proxies.append({
            'ip':   row.find_all('td')[0].string,
            'port': row.find_all('td')[1].string,
            'anonymity': row.find_all('td')[4].string
        })

    # Iteratively check all the proxies we have, and keep only the functioning ones
    working_proxies = []
    for i in tqdm(range(0,len(proxies))):
        working_proxies.append(check_proxy(proxies[i]['ip'] + ':' + proxies[i]['port']))

    # Clean of the None values our list of working proxies
    working_proxies = [proxy for proxy in working_proxies if proxy != None]
    return working_proxies