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


df = pd.read_pickle(f"{path_to_processed}final_df_clean.pkl.gz", compression = 'gzip')
df = df.loc[(df.week_start <= pd.datetime(2020, 12, 31))&(df.week_start >= pd.datetime(2020, 2, 24))]
df.drop(columns = 'tweet_text', inplace = True)

# Categorize user activity for their sentiment
df['main_emo'] = df[['anger', 'fear', 'joy', 'sadness']].idxmax(axis = 1)


# PARAMETERS -----
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
    colormap = plt.cm.get_cmap('bwr_r') # 'plasma' or 'viridis'
    plt.pcolor(corr_matrix, cmap = colormap, vmin = -1, vmax = 1)
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

    # Use STL to detrend/deseasonalize all our observations
    store = pd.DataFrame()
    for col in corr_df.columns:
        stl = STL(corr_df[col], period = period, seasonal = seasonal)
        res = stl.fit()
        store[col] = res.resid
    # Fix order of index
    store = store.T.reindex(['far_left', 'center_left', 'center', 'center_right', 'far_right']).T

    return store


# Build our dataset
store = build_resid(period, seasonal)
# Build a pearson correlation matrix
corr_matrix = store.corr(method = method)
build_heatmap()
save_fig(f"{path_to_figures_corr}pol_act_heatmap")
corr_matrix


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

# positions for all nodes - seed for reproducibility
pos = nx.spring_layout(G, seed=42) 

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
label_dict = {'far_right': "Far Right", 'far_left':'Far Left', 'center_left': 'Center Left', 'center': 'Center', 'center_right': 'Center Right'}


fig, ax = plt.subplots(1,1, figsize=(8,6))
pos = nx.nx_agraph.graphviz_layout(G, 'neato', root = 42)
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
plt.show()