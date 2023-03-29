###########################################################
# TOPIC MODELING - LDA
# Author: Luca Adorni
# Date: March 2023
###########################################################

# 0. Setup -------------------------------------------------

import re
import os
import sys
import dill
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import warnings
import snap
import itertools
import collections
from nltk import bigrams
import networkx as nx

tqdm.pandas()

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.simplefilter("ignore")

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
path_to_ctm = f"{path_to_data}ctm/"
path_to_lda = f"{path_to_data}lda/"
path_to_sk_lda = f"{path_to_data}sk_lda/"

# 1. Load our dataset --------------------------------------------

# PARAMETER ---------
stem = True
stem_tag = np.where(stem, "_stemm","")
do_mentions = False
mention_tag = np.where(do_mentions, "_mentions", "")
do_users = False
mention_tag = np.where(do_users, "_users", "")
do_switch = True
mention_tag = np.where(do_switch, "_pol_switch", "")

# Build our snap-txt compatible file
try:
    # Reload everything

    with open(f"{path_to_processed}id_network{mention_tag}.p", "rb") as outfile:
        unique_bi = pickle.load(outfile)
    print("Loaded ID dictionary")

    bigram_df = pd.read_csv(f"{path_to_processed}graph_snap{mention_tag}.txt", sep = "\t")
    print("Loaded TXT File")
except:
    print("Building network file")
    # Load our main dataset
    final_df = pd.read_pickle(f"{path_to_processed}cleaned_gsdmm_tweets{stem_tag}.pkl.gz", compression = 'gzip')
    
    # If we are doing users, keep only user-polarization pairs
    if do_users:
        final_df = final_df[['scree_name', 'polarization_bin']]
        final_df['hashtag'] = final_df.scree_name.apply(lambda x: [x])
    elif do_switch:
        final_df = final_df[['scree_name', 'week_start', 'polarization_bin']].drop_duplicates()
        # Sort by user/weekstart and get the previous polarization
        final_df.sort_values(by = ['scree_name', 'week_start'], inplace = True)
        final_df['old_pol'] = final_df.groupby(['scree_name']).polarization_bin.shift(1)

    else:
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
    
    # Create also 0/1 for being in a certain group
    final_df = pd.concat([final_df, pd.get_dummies(final_df.polarization_bin)], axis = 1)

    # Add polarization in the list of hashtags to make our network
    final_df.progress_apply(lambda x: x['hashtag'].append(x['polarization_bin']), axis = 1)

    # Create list of lists containing bigrams in tweets
    terms_bigram = [list(bigrams(tweet)) for tweet in final_df.hashtag if len(tweet) > 1]

    # Flatten list of bigrams in clean tweets
    bigram_list = list(itertools.chain(*terms_bigram))
    bigram_df = pd.DataFrame(bigram_list, columns = ['FromNodeId', 'ToNodeId'])

    # Convert to IDs

    unique_bi = bigram_df.FromNodeId.unique().tolist() + bigram_df.ToNodeId.unique().tolist()
    unique_bi = list(set(unique_bi))
    # Factorize to unique IDs everything
    unique_bi = pd.factorize(unique_bi)
    # Then make a dictionary out of it
    unique_bi = {unique_bi[1][i]:unique_bi[0][i] for i,_ in enumerate(unique_bi[0])}
    # Save the dictionary
    with open(f"{path_to_processed}id_network{mention_tag}.p", "wb") as outfile:
        pickle.dump(unique_bi, outfile)


    # Now convert our dataframe to numerical ids
    bigram_df['FromNodeId'] = bigram_df['FromNodeId'].progress_apply(lambda x: unique_bi[x])
    bigram_df['ToNodeId'] = bigram_df['ToNodeId'].progress_apply(lambda x: unique_bi[x])

    # Save everything to a txt file
    bigram_df.to_csv(f"{path_to_processed}graph_snap{mention_tag}.txt", sep = "\t", index = False, header = False)

# 2. Build our graph --------------------------------------------

# Load from our snap-txt file
G = snap.LoadEdgeList(snap.TUNGraph, f"{path_to_processed}graph_snap{mention_tag}.txt", 0, 1, Separator = "\t")
# Print some basic informations
G.PrintInfo("Hashtags", f"hashtag_info{mention_tag}.txt", False)
# Graph:
# Nodes:                    137615
# Edges:                    616924
# Zero Deg Nodes:           0
# Zero InDeg Nodes:         0
# Zero OutDeg Nodes:        0
# NonZero In-Out Deg Nodes: 137615


labels_dict = {'far_right': unique_bi['far_right'], 
               'far_left': unique_bi['far_left'],
               'center': unique_bi['center'],
               'center_left': unique_bi['center_left'],
               'center_right':unique_bi['center_right']}


store = []
for k1, v1 in labels_dict.items():
    n1 = final_df.loc[final_df[k1] == 1].shape[0]
    for k2, v2 in labels_dict.items():
        n2 = final_df.loc[final_df[k2] == 1].shape[0]
        # Get neighbors
        NodeVec = snap.TIntV()
        n_nodes1 = snap.GetNodesAtHop(G, int(v1), 1, NodeVec, False) 
        n_nodes2 = snap.GetNodesAtHop(G, int(v2), 1, NodeVec, False) 
        # Common Neighbors
        comm = G.GetCmnNbrs(int(v1), int(v2))
        print(f"{k1} and {k2}: {comm/n_nodes1}")

        print(f"NOVER: {k1} and {k2}: {comm/(n_nodes1 + n_nodes2 - 2 - comm)}")
        store.append((k1, k2, comm, n_nodes1, n_nodes2, n1, n2))
    
comm_df = pd.DataFrame(store, columns = ['orig', 'direc', 'comm', 'n_nodes1', 'n_nodes2', 'n_orig', 'n_direc'])
# Remove the 100%
comm_df = comm_df.loc[comm_df.orig != comm_df.direc]
comm_df['perc'] = comm_df.comm/comm_df.n_nodes1
comm_df['perc_act'] = comm_df.n_orig/comm_df.n_direc

comm_df['perc_act'] = comm_df.n_nodes1/comm_df.n_nodes2

comm_df['perc2'] = comm_df.perc*comm_df.perc_act

comm_df['prov'] = comm_df.n_nodes1/comm_df.n_orig
comm_df['prov2'] = comm_df.n_nodes2/comm_df.n_direc

comm_df['perc_act'] = comm_df.prov2/comm_df.prov

comm_df['order'] = 0
comm_df.loc[comm_df.direc == 'far_right', 'order'] = 1
comm_df.loc[comm_df.direc == "center_right", 'order'] = 0.5
comm_df.loc[comm_df.direc == "center_left", 'order'] = -0.5
comm_df.loc[comm_df.direc == "far_left", 'order'] = -1
comm_df.sort_values(by = 'order', inplace = True)

comm_df['nover'] = comm_df.comm/(comm_df.n_nodes1 + comm_df.n_nodes2 - 2 - comm_df.comm)
# https://www.mdpi.com/1999-4893/9/1/8
import seaborn as sns
import matplotlib.pyplot as plt

sns.lineplot(data = comm_df, x = 'direc', y = 'nover', hue = 'orig')
plt.show()

# FAR RIGHT vs FAR LEFT
G.GetCmnNbrs(15125, 56598)
# FAR RIGHT VS CENTER LEFT
G.GetCmnNbrs(15125, 55951)
# FAR RIGHT VS CENTER 
G.GetCmnNbrs(15125, 115813)
# FAR RIGHT VS CENTER RIGHT
G.GetCmnNbrs(15125, 100921)


G.GetNodeClustCf(4089)
G.GetNodeClustCf(104705)

G.GetDegreeCentr(4089)

G.GetDegreeCentr(126811)

G.Nodes(15125)


NodeVec = snap.TIntV()
snap.GetNodesAtHop(G, 15125, 1, NodeVec, False) 


NodeVec = snap.TIntV()
snap.GetNodesAtHop(G, 104705, 1, NodeVec, False) 



snap.DrawGViz(G, snap.gvlDot, f"G1{mention_tag}.png", "G1")