###########################################################
# Code to print words for topic-author models
# Author: Luca Adorni
# Date: May 2023
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


pd.options.display.max_columns = 200
pd.options.display.max_rows = 1000
pd.set_option('max_info_columns', 200)
pd.set_option('expand_frame_repr', False)
pd.set_option('expand_frame_repr', True)
pd.set_option('max_colwidth',1000)
pd.set_option('display.width',None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


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
path_to_author = f"{path_to_data}author/"

os.makedirs(path_to_ctm, exist_ok=True)
os.makedirs(path_to_lda, exist_ok=True)
os.makedirs(path_to_gsdmm, exist_ok=True)
os.makedirs(path_to_author, exist_ok=True)

# 1. LOAD DATA -------------------------------------------------

# Load the labelled author-topic model data
df = pd.read_csv(f"{path_to_author}final/resulting_topics_periods_final.csv")

# Remove columns we do not need
df.drop(columns = 'types', inplace = True)

# Rename variables
df.month.replace({'first_lock': 'First Lockdown', 'post_lock': "End of Lockdown", 
                 'summer': 'Summer', 'second_lockdown': 'Second Lockdown'}, inplace = True)

# Now iterate over the top words and clean them out

def format_topwords(top_words):
    # Define beginning and end latex commands
    beg = "\\begin{tabular}[c]{@{}l@{}}"
    end = "\\end{tabular}"
    # split into a list
    top_words = top_words.split(",")
    top_words = [re.search('"(.*)"', x).group(0) for x in top_words]
    top_words = [re.sub('"', "", x) for x in top_words]
    top_words = [re.sub('_','\\_', x) for x in top_words]

    # Now, every 10 tokens go to a new line
    top_words[10] = "\\\\" + top_words[10]
    top_words[20] = "\\\\" + top_words[20] 
    # Add beginning and end to the string
    top_words[0] = beg + top_words[0]
    top_words[29] = top_words[29] + end
    top_words = ",".join(top_words)
    return top_words

df['top_words'] = df.top_words.apply(lambda x: format_topwords(x))

# Change positions
df['topic_n'] = df.groupby('month').cumcount()+1
df = df[['topic_n', 'month', 'macro', 'far_left', 'center_left', 'center', 'center_right', 'far_right', 'top_words']]
df.rename(columns = {'far_left':'Far Left', 'center_left':"Center Left", 'center':'Center', 
                    'center_right': 'Center Right', 'far_right':"Far Right", 'macro': 'Category', 'month': 'Period',
                     'top_words': 'Top Words',
                    'topic_n': 'Topic'}, inplace = True)

# Split into two datasets
df1 = df.loc[df.Period.isin(['First Lockdown', 'End of Lockdown'])]
df2 = df.loc[df.Period.isin(['First Lockdown', 'End of Lockdown']) == False]


df1.to_latex(f"{path_to_tables}table_1_author_lda_words.tex", escape = False, index = False, header = True)
df2.to_latex(f"{path_to_tables}table_2_author_lda_words.tex", escape = False, index = False, header = True)