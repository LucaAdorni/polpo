###########################################################
# Code to create plot graphs
# Author: Luca Adorni
# Date: April 2023
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
path_to_figures_final = f"{path_to_figures}final/"


# 1. Load Dataset ----------------------------------------------------

def build_heatmap():
    # Plot a heatmap of all the correlation coefficients
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    colormap = plt.cm.get_cmap('Blues') # 'plasma' or 'viridis'
    plt.pcolor(corr_matrix, cmap = colormap, vmin = 0, vmax = 1)
    plt.colorbar(cmap = colormap)
    ax = plt.gca()
    ax.set_xticks(np.arange(corr_matrix.shape[0])+0.5, fontsize = 30)
    ax.set_yticks(np.arange(corr_matrix.shape[0])+0.5, fontsize = 30)
    ax.set_xticklabels(corr_matrix.columns, rotation=80, fontsize = 15)
    ax.set_yticklabels(corr_matrix.columns, fontsize = 15)
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


corr_matrix = pd.read_pickle(f"{path_to_figures_corr}corr_matrix.pkl.gz", compression = 'gzip')


# 2. HEATMAP ----------------------------------------------------

# Build a heatmap
build_heatmap()
save_fig(f"{path_to_figures_final}pol_act_heatmap")

# 3. INFERRED NETWORK ----------------------------------------------------

# Threshold to plot correlations
threshold = 0.49

# Get our list of edges with their relative weights
edge_list = []
for i in corr_matrix.index:
        for j in corr_matrix.columns:
            if round(corr_matrix.loc[i,j],2) >= threshold and i != j:
                edge_list.append((j,i,corr_matrix.loc[i,j]))

# Initialize our weighted graph
G = nx.Graph()

# Iterate over all edges and add them
for edge in edge_list:
    G.add_edge(edge[0], edge[1], weight=edge[2])


# Define a dictionary of labels
label_dict = {'far_right': "Far Right", 'far_left':'Far Left', 
              'center_left': 'Center Left',
            'center': 'Center', 'center_right': 'Center Right', 'anti-gov': "Anti-Gov",
            'lombardia': "Lombardia", 'immuni': 'Immuni', 'fake-news':'Fake-News', 'pro-lock': 'Pro-Lock'}




def nudge(pos, x_shift, y_shift):
    return {n:(x + x_shift, y + y_shift) for n,(x,y) in pos.items()}

# positions for all nodes - seed for reproducibility
pos = nx.spring_layout(G, seed=42) 
# Nudge the labels
pos_nodes = nudge(pos, 0, 0.1)        


fig, ax = plt.subplots(1,1, figsize=(8,6))

# Node labels (nudged)
nx.draw_networkx_labels(G, pos=pos_nodes, ax=ax)         # nudged labels
# Set options to get colored edges
options = {
    "edge_color": [d['weight'] for (u, v, d) in G.edges(data=True)],
    'width': [d['weight']*6 for (u, v, d) in G.edges(data=True)],
    "edge_cmap": plt.cm.Blues,
    "with_labels": False,
    'node_color': "skyblue",
}

# Add additional edge color (to make more visible the transparent ones)
for edge in G.edges(data='weight'):
    nx.draw_networkx_edges(G, pos, edgelist=[edge], width=edge[2]*7)
# Draw the plot
nx.draw(G, pos, **options, ax = ax)
ax = plt.gca()
ax.margins(0.08)
plt.axis("off")
plt.tight_layout()
save_fig(f"{path_to_figures_corr}pol_act_netw")
plt.show()