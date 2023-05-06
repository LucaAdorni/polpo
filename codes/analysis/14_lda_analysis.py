###########################################################
# TOPIC MODELING - LDA
# Author: Luca Adorni
# Date: March 2023
###########################################################

# 0. Setup -------------------------------------------------

import re
import os
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import dill
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import warnings
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
path_to_sk_lda = f"{path_to_data}sk_lda/"
path_to_figures_final = f"{path_to_figures}final/"

os.makedirs(path_to_sk_lda, exist_ok=True)

# 1. Load our dataset --------------------------------------------

# PARAMETER ---------
stem = True
stem_tag = np.where(stem, "_stemm","")
period = 'periods'
if period == 'months':
    # Do LDA for the most relevant weeks during the pandemic
    week_dict = {'feb': [2], 'mar': [3], 'apr': [4], 'may': [5], 
                'june': [6], 'jul': [7],'aug': [8], 'sep': [9], 
                'oct': [10],'nov': [11], 'dec': [12]}

    n_topics = 10
elif period == 'periods':
    # Do LDA for the most relevant weeks during the pandemic
    week_dict = {
                'second_all': [7,8,9,10,11,12],
                'middle': [5,6,7,8,9],
                'all':[2,3,4,5,6,7,8,9,10,11,12],
                'end_lock': [5,6],
                'summer': [7,8,9], 
                'second_lockdown': [10,11,12],
                'first_lockdown': [2,3,4]}
    n_topics = 20

final_df = pd.read_pickle(f"{path_to_processed}cleaned_gsdmm_tweets{stem_tag}.pkl.gz", compression = 'gzip')
# We want to have who switch vs who doesn't
final_df['polarization_bin'] = np.where(final_df.extremism_toleft == 1, "extremism_toleft",
                                        np.where(final_df.extremism_toright == 1, 'extremism_toright',
                                        np.where(final_df.orient_change_toleft == 1, 'orient_change_toleft',
                                        np.where(final_df.orient_change_toright == 1, 'orient_change_toright',
                                        final_df.polarization_bin))))


# Create also 0/1 for being in a certain group
final_df = pd.concat([final_df, pd.get_dummies(final_df.polarization_bin)], axis = 1)

# Define two lists for the different polarization measures we have
pol_changes = ["extremism_toright",'center_right', "extremism_toleft","orient_change_toleft",
            "orient_change_toright",'far_left', 'far_right', 'center_left', 
                 'center']

# 2. FUNCTIONS --------------------------------------------

def vectorize_to_dataframe(df, vectorizer_obj):
    """
    Function to return a dataframe from our vectorizer results
    """
    df = pd.DataFrame(data = df.toarray(), columns = vectorizer_obj.get_feature_names())
    return df


def display_topics(model, feature_names, no_top_words):
    """ Function to print all the main words from each topic """
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        print("-"*10)

def top_tweets(model, topic_dct, feature_names, no_top_words):
    """ Function to get a dataframe with each topic, its main words and an example tweet"""
    topic_store = []
    tweet_store = []
    word_store = []
    for topic_idx, topic in enumerate(model.components_):
        topic_store.append(f"topic_{topic_idx}")
        tweet_store.append(df.iloc[topic_dct[f'topic_{topic_idx}'].idxmax(axis = 0)].tweet_text)
        word_store.append([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
    return pd.DataFrame([topic_store, word_store, tweet_store], index = ['Topic','Words', 'Tweets']).T

# define a function that saves figures
def save_fig(fig_id, tight_layout=True):
    # The path of the figures folder ./Figures/fig_id.png (fig_id is a variable that you specify 
    # when you call the function)
    path = os.path.join(fig_id + ".pdf") 
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='pdf', dpi=300)

# 3. LDA -----------------------------------------------------------------------------------------------------------

# ALL --------------------------------------------------------------------------------------------------------------

# PARAMETERS ----------------------
key_w = 'all'
week_date = week_dict[key_w]
pol = 'extremism_toright'
tag = '_pol'
# ----------------------------------

# Subset our dataframe
sub_df = final_df.loc[(final_df.week_start.dt.month.isin(week_date))&(final_df['polarization_bin'] == pol)]
# Train a topic for politic tweets
pol_df = sub_df.loc[(sub_df.topic == 'politics')|(sub_df.topic == 'economics')]
# And for health tweets
vac_df = sub_df.loc[(sub_df.topic == 'health')|(sub_df.topic == 'vaccine')]
iter_dict = {'_vacc': vac_df, '_pol': pol_df}

# Clean our dataframe and compute our DTM
df = iter_dict[tag].copy()
vectorizer = CountVectorizer(ngram_range=(1,2), min_df=5)
df_token = vectorizer.fit_transform(df.tokens)
df_token = vectorize_to_dataframe(df_token, vectorizer)

try:
    with open(f'{path_to_sk_lda}lda_{key_w}{tag}{pol}{stem_tag}_{n_topics}', 'rb') as file: # and save the fitted model
        lda = dill.load(file)
        print("Model already trained")
except:
    print("Training LDA")
    lda = LatentDirichletAllocation(n_components = n_topics, random_state = 42, n_jobs = -1)
    lda.fit(df_token)
    with open(f'{path_to_sk_lda}lda_{key_w}{tag}{pol}{stem_tag}_{n_topics}', 'wb') as file: # and save the fitted model
        dill.dump(lda, file)

# Extract its components
lda_components = pd.DataFrame(lda.components_, columns = df_token.columns).T
lda_components.to_csv(f'{path_to_sk_lda}lda_{key_w}{tag}{pol}{stem_tag}{n_topics}.csv')

# Print its top words
no_top_words = 20
display_topics(lda, vectorizer.get_feature_names(), no_top_words)

# Create a DocumentXTopic Matrix
topic_dct = lda.transform(df_token)
topic_dct = pd.DataFrame(topic_dct, columns = [f"topic_{i}" for i in range(n_topics)])
# Get the predicted topic
topic_dct['pred'] = topic_dct.idxmax(axis = 1)
df['pred'] = topic_dct['pred'].tolist()


# Get the Top Tweets
top_df = top_tweets(lda, topic_dct, vectorizer.get_feature_names(), no_top_words)


# 0: Covid-19 and migrants - ok
# 1: Migrants - ok
# 2: Government + Meloni + China - ok
# 3: dictatorship - ok
# 4: Other/empty - ok
# 5: Anti-lockdown - ok
# 6: Process against Conte - ok
# 7: China supplying respirators (other) - ok
# 8: Abuse of power - ok
# 9: Secret documents of the government - ok
# 10: Against excessive fear of Covid - ok
# 11: Against government - ok
# 12: MES + Against government - ok
# 13: Against government - ok
# 14: Against government + Migrants - ok
# 15: Against Arcuri (government) - ok
# 16: Against chinese/wuhan
# 17: Secret documents of the government - ok
# 18: Against government - ok
# 19: Covid created in a lab - ok

# Create macro-topics
topic_dict = {'Xenophobia': [0, 1, 14, 16],
              'Anti-Government': [2, 5, 10, 11, 12, 13, 15, 18],
              'Conspiracies': [3, 6, 8, 9, 17, 19],
              'Other': [4, 7]}


for k,v in topic_dict.items():
    topic_dict[k] = [f"topic_{i}" for i in v]

df['macro'] = ""
top_df['Macro-Topic'] = ""
for k in topic_dict.keys():
    df.loc[df.pred.isin(topic_dict[k]), 'macro'] = k
    top_df.loc[top_df.Topic.isin(topic_dict[k]), 'Macro-Topic'] = k

# Cleanup and export to LaTex
top_df = top_df[['Topic', 'Macro-Topic', 'Words', 'Tweets']]
top_df['Topic'] = top_df['Topic'].apply(lambda x: re.sub("topic_", "Topic ", x))
top_df['Words'] = top_df['Words'].apply(lambda x: ", ".join(x))
top_df.to_latex(f"{path_to_sk_lda}clean/table_a3_top_tweets{key_w}{tag}{pol}{stem_tag}{n_topics}.tex")

# Now generate the plot over time
df['n_obs'] = 1
plot_df = df.groupby(['week_start', 'macro'], as_index = False).n_obs.sum()
plot_df['tot'] = plot_df.groupby('week_start').n_obs.transform('sum')
plot_df['n_obs'] = plot_df.n_obs/plot_df.tot
plot_df = plot_df.loc[plot_df.macro != 'Other']

# Balance our panel data
# Get Unique macro-topics
uid = plot_df['macro'].unique()
# Get an array of week_start and transform it to a dataframe
dates = np.array([plot_df.week_start.unique() for x in range(len(uid))])
balanced_panel = pd.DataFrame(dates).T
balanced_panel.columns = uid
# Bring everything to long format and merge it back
balanced_panel = pd.melt(balanced_panel, var_name = 'macro', value_name = 'week_start')
rebalanced_data = pd.merge(balanced_panel, plot_df, how='left', on=['macro', 'week_start']).fillna(0)
# Smooth the series using MA(2)
rebalanced_data['n_obs'] = rebalanced_data.groupby(['macro'], as_index = False).n_obs.transform(lambda x: x.rolling(window = 3).mean())

# Plot over time the macro-topics
fig, ax = plt.subplots(figsize=(15, 10))
sns.lineplot(data = rebalanced_data, x = 'week_start', y = 'n_obs', 
             hue = 'macro', markers = True,
             markersize = 10, style = 'macro', dashes = True, ax = ax)
sns.despine()
plt.ylabel('% of Tweets', fontsize = 35)
plt.xlabel('Weeks', fontsize = 35)
plt.yticks(fontsize = 30)
plt.xticks(fontsize = 30)
plt.legend(loc = 'upper right', fontsize = 35, ncol = 2, frameon=False)
plt.ylim(0,1)
ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

save_fig(f'{path_to_figures_final}fig_4_{key_w}{tag}{pol}{stem_tag}{n_topics}')

# ALL --------------------------------------------------------------------------------------------------------------

# PARAMETERS ----------------------
key_w = 'all'
week_date = week_dict[key_w]
pol = 'extremism_toright'
tag = '_vacc'
# ----------------------------------

# Clean our dataframe and compute our DTM
df = iter_dict[tag].copy()
vectorizer = CountVectorizer(ngram_range=(1,2), min_df=5)
df_token = vectorizer.fit_transform(df.tokens)
df_token = vectorize_to_dataframe(df_token, vectorizer)

try:
    with open(f'{path_to_sk_lda}lda_{key_w}{tag}{pol}{stem_tag}_{n_topics}', 'rb') as file: # and save the fitted model
        lda = dill.load(file)
        print("Model already trained")
except:
    print("Training LDA")
    lda = LatentDirichletAllocation(n_components = n_topics, random_state = 42, n_jobs = -1)
    lda.fit(df_token)
    with open(f'{path_to_sk_lda}lda_{key_w}{tag}{pol}{stem_tag}_{n_topics}', 'wb') as file: # and save the fitted model
        dill.dump(lda, file)

# Extract its components
lda_components = pd.DataFrame(lda.components_, columns = df_token.columns).T
lda_components.to_csv(f'{path_to_sk_lda}lda_{key_w}{tag}{pol}{stem_tag}{n_topics}.csv')

# Print its top words
no_top_words = 20
display_topics(lda, vectorizer.get_feature_names(), no_top_words)

# Create a DocumentXTopic Matrix
topic_dct = lda.transform(df_token)
topic_dct = pd.DataFrame(topic_dct, columns = [f"topic_{i}" for i in range(n_topics)])
# Get the predicted topic
topic_dct['pred'] = topic_dct.idxmax(axis = 1)
df['pred'] = topic_dct['pred'].tolist()


# Get the Top Tweets
top_df = top_tweets(lda, topic_dct, vectorizer.get_feature_names(), no_top_words)

# 0: Spreading fear
# 1: Mismanagement from doctors
# 2: Conspiracies (Wuhan + Gates)
# 3: New strains
# 4: Measures
# 5: Migrants
# 6: Migrants
# 7: Vaccines
# 8: Measures
# 9: Measures
# 10: Cases
# 11: Some conspiracies tweets
# 12: Cases
# 13: Something against lockdown etc.
# 14: Migrants
# 15: Cases
# 16: Conspiracies
# 17: Cases
# 18: Cases
# 19: Mismanagement

i = 19
df.loc[df.pred == f"topic_{i}"].tweet_text
top_df.loc[top_df.Topic == f"topic_{i}"]


# Create macro-topics
topic_dict = {'Xenophobia': [5, 6, 14],
              'Conspiracies': [2, 11, 16],
              'Lockdown': [1, 19, 0],
              'Other': [3, 4, 7, 8, 9, 10, 12, 13, 15, 17, 18]}

for k,v in topic_dict.items():
    topic_dict[k] = [f"topic_{i}" for i in v]

df['macro'] = ""
top_df['Macro-Topic'] = ""
for k in topic_dict.keys():
    df.loc[df.pred.isin(topic_dict[k]), 'macro'] = k
    top_df.loc[top_df.Topic.isin(topic_dict[k]), 'Macro-Topic'] = k

# Cleanup and export to LaTex
top_df = top_df[['Topic', 'Macro-Topic', 'Words', 'Tweets']]
top_df['Topic'] = top_df['Topic'].apply(lambda x: re.sub("topic_", "Topic ", x))
top_df['Words'] = top_df['Words'].apply(lambda x: ", ".join(x))
top_df.to_latex(f"{path_to_sk_lda}clean/table_a4_top_tweets{key_w}{tag}{pol}{stem_tag}{n_topics}.tex")

# Now generate the plot over time
df['n_obs'] = 1
plot_df = df.groupby(['week_start', 'macro'], as_index = False).n_obs.sum()
plot_df['tot'] = plot_df.groupby('week_start').n_obs.transform('sum')
plot_df['n_obs'] = plot_df.n_obs/plot_df.tot
plot_df = plot_df.loc[plot_df.macro != 'Other']

# Balance our panel data
# Get Unique macro-topics
uid = plot_df['macro'].unique()
# Get an array of week_start and transform it to a dataframe
dates = np.array([plot_df.week_start.unique() for x in range(len(uid))])
balanced_panel = pd.DataFrame(dates).T
balanced_panel.columns = uid
# Bring everything to long format and merge it back
balanced_panel = pd.melt(balanced_panel, var_name = 'macro', value_name = 'week_start')
rebalanced_data = pd.merge(balanced_panel, plot_df, how='left', on=['macro', 'week_start']).fillna(0)
# Smooth the series using MA(2)
rebalanced_data['n_obs'] = rebalanced_data.groupby(['macro'], as_index = False).n_obs.transform(lambda x: x.rolling(window = 3).mean())

# Plot over time the macro-topics
fig, ax = plt.subplots(figsize=(15, 10))
sns.lineplot(data = rebalanced_data, x = 'week_start', y = 'n_obs', 
             hue = 'macro', markers = True,
             markersize = 10, style = 'macro', dashes = True, ax = ax)
sns.despine()
plt.ylabel('% of Tweets', fontsize = 35)
plt.xlabel('Weeks', fontsize = 35)
plt.yticks(fontsize = 30)
plt.xticks(fontsize = 30)
plt.legend(loc = 'upper right', fontsize = 35, ncol = 2, frameon=False)
plt.ylim(0,1)
ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

save_fig(f'{path_to_figures_final}fig_4_{key_w}{tag}{pol}{stem_tag}{n_topics}')