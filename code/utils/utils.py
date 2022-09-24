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
    now = time.time()

    def unshorten_url(url):
        """
        Function to unshorten a single URL
        """
        try: r = session.head(url, allow_redirects=True, timeout = 10).url
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