###########################################################
# Code to create plot graphs
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
from collections import Counter
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import random
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.seasonal import STL
import networkx as nx
from tqdm import tqdm

tqdm.pandas()

import sys
sys.path.append("/Users/ADORNI/Documents/graph-tool/src")

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


path_to_figures_corr = f"{path_to_figures}corr_heat/"

os.makedirs(path_to_figures_corr, exist_ok = True)

# 1. LOAD DATASET ------------------------------------



# PARAMETERS ---------
stem = True
stem_tag = np.where(stem, "_stemm","")
# If do_mentions = True, extract mentions instead of hashtags
do_mentions = False
# Set to True if we want to count just once a tweet if a hashtag appears, otherwise count the number of hashtag appearances
count_hash_app = False
# Set to True if we want to check the correlation in n. of tweets
pol_activity = False
# PARAMETER: Correlation method
method = 'spearman'
# STL Smoothing parameters
period = 2
seasonal = 7
# -----------------

def build_heatmap():
    # Plot a heatmap of all the correlation coefficients
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    colormap = plt.cm.get_cmap('Blues') # 'plasma' or 'viridis'
    plt.pcolor(corr_matrix, cmap = colormap, vmin = 0, vmax = 1)
    plt.colorbar(cmap = colormap)
    ax = plt.gca()
    ax.set_xticks(np.arange(corr_matrix.shape[0])+0.5)
    ax.set_yticks(np.arange(corr_matrix.shape[0])+0.5)
    ax.set_xticklabels(corr_matrix.columns, rotation=80)
    ax.set_yticklabels(corr_matrix.columns)
    ax.set_aspect('equal')

# define a function that saves figures
def save_fig(fig_id, tight_layout=True):
    # The path of the figures folder ./Figures/fig_id.pdf (fig_id is a variable that you specify 
    # when you call the function)
    path = os.path.join(fig_id + ".pdf") 
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='pdf', dpi=300)


def build_resid(period = 44, seasonal = 5):
    # Get the weekly-activity for each:
    # a. polarization group
    if pol_activity:
        corr_df = df.groupby(['week_start', 'polarization_bin'], as_index = False).n_tweets.sum()
        corr_df = corr_df.pivot(columns = ['polarization_bin'], index = 'week_start', values = 'n_tweets')
    else:
        corr_df = df.groupby(['week_start', 'polarization_bin'], as_index = False).scree_name.count()
        corr_df = corr_df.pivot(columns = ['polarization_bin'], index = 'week_start', values = 'scree_name')
    hashtag_df = final_df.groupby(['week_start'], as_index = True)[[k for k in hash_dict.keys()]].sum()

    corr_df = pd.concat([corr_df, hashtag_df], axis = 1)

    # Use STL to detrend/deseasonalize all our observations
    store = pd.DataFrame()
    for col in corr_df.columns:
        stl = STL(corr_df[col], period = period, seasonal = seasonal)
        res = stl.fit()
        store[col] = res.resid
    # Fix order of index
    store = store.T.reindex(['far_left', 'center_left', 'center', 'center_right', 'far_right', 'anti-gov', 'pro-lock', 'china', 'immuni',
                             'conspiracy', 'novax', 'migrants', 'fake-news',
                             ]).T

    return store


# Load User/Polarization dataset
df = pd.read_pickle(f"{path_to_processed}final_df_clean.pkl.gz", compression = 'gzip')
df = df.loc[(df.week_start <= pd.datetime(2020, 12, 31))&(df.week_start >= pd.datetime(2020, 2, 24))]
df.drop(columns = 'tweet_text', inplace = True)

# Categorize user activity for their sentiment
df['main_emo'] = df[['anger', 'fear', 'joy', 'sadness']].idxmax(axis = 1)

# Load our main dataset
final_df = pd.read_pickle(f"{path_to_processed}cleaned_gsdmm_tweets{stem_tag}.pkl.gz", compression = 'gzip')
# Restrict the time
final_df = final_df.loc[(final_df.week_start <= pd.datetime(2020, 12, 31))&(final_df.week_start >= pd.datetime(2020, 2, 24))]

# Search for all hashtags
def get_hashtag(x):
    # Lowercase everything
    x = x.lower()
    # We first remove any trace of covid
    x = re.sub('(coronavirus|covid19|covid-19|covid19italia|coronavid19|pandemia|corona|virus|covid|covid_19)', '', x)  # remove hash tags
    if do_mentions:
        return re.findall("@[A-Za-z]+[A-Za-z0-9-_]+", x)
    else:
        return re.findall("#[A-Za-z]+[A-Za-z0-9-_]+",x)

# Extract only relevant hashtags
final_df['hashtag'] = final_df.tweet_text.progress_apply(lambda x: get_hashtag(x))

# Get the top hashtags
hashtags = final_df.hashtag.tolist()

# Flatten it
hashtags = [l for lis in hashtags for l in lis]
hashtags = pd.DataFrame(hashtags, columns = ['hashtags'])

# Group hashtags per topics
hash_dict = {'anti-gov': [h for h in  hashtags.hashtags.unique().tolist() if 
                            (re.search("governo|conte|speranza|pd|m5s", h) 
                            and re.search("criminal|vergogn|dimett|irresp|merd|infam|acasa|cazzo|c4zz0", h)) 
                            or (re.search("pdiot|pidiot", h)) or (re.search("dittatura", h))],
             'pro-lock': [h for h in  hashtags.hashtags.unique().tolist() if (re.search("casa", h) 
                            and re.search("stare|sto|stiamo|resto|restare|restiamo|rimanere|rimango|rimaniamo", h) 
                            and re.search("cazzo|non|rotto|odio|accidenti|beata|nn|c4zz0|mah", h) == None) 
                            or (re.search("andra|andare", h) 
                            and re.search("bene", h) and re.search("sega|cazzo|non|accidenti|beata|nn|c4zz0|mah", h) == None)
                            or (re.search("uniti", h) and (re.search("distanti|restiamo|restare", h)))],
             'immuni': [h for h in  hashtags.hashtags.unique().tolist() if (re.search("immuni", h)
                    and re.search("immuniz|immunit", h) == None)
                    or (re.search('privacy|tracing|tracciamento', h))],
             'china': ['#cina', '#wuhan', '#china', '#cinesi', '#cinese', '#chinese']+[h for h in  hashtags.hashtags.unique().tolist() if re.search("wuhan", h)],
             'fake-news': [h for h in  hashtags.hashtags.unique().tolist() if re.search("fakenew(s)|disinformazione|infodemia", h)],
             'conspiracy': [h for h in  hashtags.hashtags.unique().tolist() if re.search("gates|soros", h)],
             'novax': [h for h in  hashtags.hashtags.unique().tolist() if (re.search("vaccin|pfizer|biontech|astrazeneca|vax", h) 
                                                                        and re.search("mai|fraud|stop|follia|killer|lobby", h))
                                                                        or (re.search("novaccin|noalvaccin", h))
                                                                        or (re.search("novax|antivax|no-vax", h))],
             'migrants':[h for h in  hashtags.hashtags.unique().tolist() if (re.search("porto|porti", h) and re.search("chiud|chius", h))
                                                        or (re.search('migrant|immigra|lampedusa|lamorgese|extracomunit|barcon|sbarc|clandestin', h))]
}
                         
                         
                         
                        




# TO-DO:
# Anti-Gov
# Vaccines?
# Other hashtags?


# TO DO:
# - FakeNews was correlated with leftist, check again what went wrong
# 


# Now flag all the tweets with at least one of those hashtags
for col, v in hash_dict.items():
    if count_hash_app:
        final_df[col] = final_df.hashtag.progress_apply(lambda x: bool(set(x) & set(v))).astype(int)
    else:
        final_df[col] = final_df.hashtag.progress_apply(lambda x: sum(el in x for el in v))


# Now flag all the tweets with at least one of those hashtags
for col, v in hash_dict.items():
    final_df[col] = final_df.hashtag.progress_apply(lambda x: sum(el in x for el in v))
    col = 'migrants'
    v = [h for h in  hashtags.hashtags.unique().tolist() if (re.search("porto|porti", h) and re.search("chiud|chius", h))
                                                        or (re.search('migrant|immigra|lampedusa|lamorgese|extracomunit|barcon|sbarc|clandestin', h))]

    v = v  + [h for h in  hashtags.hashtags.unique().tolist() if (re.search("casa", h) 
                and re.search("stare|sto|stiamo|resto|restare|restiamo|rimanere|rimango|rimaniamo", h) 
                and re.search("cazzo|non|rotto|odio|accidenti|beata|nn|c4zz0|mah", h)) 
                or (re.search("andra|andare", h) 
                and re.search("bene", h) and re.search("sega|cazzo|non|accidenti|beata|nn|c4zz0|mah", h))]
    final_df[col] = final_df.hashtag.progress_apply(lambda x: sum(el in x for el in v))
    final_df[col] = final_df.hashtag.progress_apply(lambda x: bool(set(x) & set(v))).astype(int)





# Build our dataset
store = build_resid(period, seasonal)
# Build a pearson correlation matrix
corr_matrix = store.corr(method = method)
corr_matrix
build_heatmap()
save_fig(f"{path_to_figures_corr}pol_act_heatmap")
plt.show()

# Threshold to plot correlations
threshold = 0.49

# Get our list of edges with their relative weights
edge_list = []
for i in corr_matrix.index:
        for j in corr_matrix.columns:
            if corr_matrix.loc[i,j] > threshold and i != j:
                edge_list.append((j,i,corr_matrix.loc[i,j]))


# Initialize our weighted graph
G = nx.Graph()

# Iterate over all edges and add them
for edge in edge_list:
    G.add_edge(edge[0], edge[1], weight=edge[2])


# elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.49]

# # nodes
# nx.draw_networkx_nodes(G, pos, node_size=700)

# # edges
# nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)

# edge weight labels
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels)

ax = plt.gca()
ax.margins(0.08)
plt.axis("off")
plt.tight_layout()
plt.show()


# Define a dictionary of labels
label_dict = {'far_right': "Far Right", 'far_left':'Far Left', 'center_left': 'Center Left',
            'center': 'Center', 'center_right': 'Center Right', 'anti-gov': "Anti-Gov",
            'china': "China", 'immuni': 'Immuni', 'fake-news':'Fake-News', 'pro-lock': 'Pro-Lock'}


fig, ax = plt.subplots(1,1, figsize=(8,6))

# positions for all nodes - seed for reproducibility
pos = nx.spring_layout(G, seed=42) 
# pos = nx.nx_agraph.graphviz_layout(G, 'neato', root = 42)
# node labels
nx.draw_networkx_labels(G, pos, labels = label_dict, font_size=20, font_family="sans-serif")
# Set options to get colored edges
options = {
    "node_color": "#A0CBE2",
    "edge_color": [d['weight'] for (u, v, d) in G.edges(data=True)],
    "width": 4,
    "edge_cmap": plt.cm.Blues,
    "with_labels": False,
}
nx.draw(G, pos, **options)
ax = plt.gca()
ax.margins(0.08)
plt.axis("off")
plt.tight_layout()
save_fig(f"{path_to_figures_corr}pol_act_netw")
plt.show()