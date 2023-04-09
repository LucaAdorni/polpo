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
path_to_tables = f"{path_to_repo}tables/"

# 1. Load Dataset ----------------------------------------------------


# Load our dataset
df = pd.read_pickle(f"{path_to_processed}final_df.pkl.gz", compression = 'gzip')
df.rename(columns = {'pred': 'prediction'}, inplace=True)
print(f"Final n. of observations: {df.shape[0]}")
print(f"Final n. of users: {df.scree_name.nunique()}")
# Final n. of observations: 1652271
# Final n. of users: 334711

# Restrict to users who have posted at least once in the pre-covid period
df['min_date'] = df.groupby('scree_name').week_start.transform('min')
df['max_date'] = df.groupby('scree_name').week_start.transform('max')
df['check'] = (df.min_date <= pd.datetime(2020, 2, 18))& (df.max_date >= pd.datetime(2020, 2, 18))
print(f"Number of observations pre-post covid:\n{df.check.value_counts()}")
print(f"Number of users pre-post covid: {df.loc[df.check].scree_name.nunique()}")
# Number of observations pre-post covid:
# True     1196826
# False     455610
# Name: check, dtype: int64
# Number of users pre-post covid: 133638

# Create a polarization variable
# df['polarization_bin'] = pd.cut(x=df['prediction'], bins = [-1.01, -0.5, -0.1, 0.1, 0.5, 1.01], labels = ["far_left", "center_left", "center","center_right", "far_right"])

def discr_bin(x):
    if x < -0.5:
        return 'far_left'
    elif x < -0.1:
        return 'center_left'
    elif x <= 0.1:
        return 'center'
    elif x <= 0.5:
        return 'center_right'
    else:
        return 'far_right'

df['polarization_bin'] = df.prediction.apply(lambda x: discr_bin(x))

print(f"Polarization bins:\n{df.polarization_bin.value_counts()}")
# Polarization bins:
# center_left     1072287
# center           374793
# center_right     132105
# far_left          71197
# far_right          1889
# Name: polarization_bin, dtype: int64
print(f"Polarization bins:\n{df.loc[df.check].polarization_bin.value_counts()}")
# Polarization bins:
# center_left     777891
# center          269446
# center_right     95969
# far_left         51988
# far_right         1413
# Name: polarization_bin, dtype: int64

# Get the user leaning
df['leaning'] = pd.cut(x=df['prediction'], bins = [-1.01, -0.1, 0.1,1.01], labels = ["left", "center","right"])
print(f"Leaning bins:\n{df.leaning.value_counts()}")
# Leaning bins:
# left      1143484
# center     374793
# right      133994
# Name: leaning, dtype: int64
print(f"Polarization bins:\n{df.loc[df.check].leaning.value_counts()}")
# Polarization bins:
# left      829879
# center    269446
# right      97382
# Name: leaning, dtype: int64


# Get the previous polarization, then study the changes direction
df = df.sort_values(by=["scree_name", "week_start"])
df['pol_old'] = df.prediction.diff()
df['bin_old'] = df.polarization_bin.shift(1)
df['leaning_old'] = df.leaning.shift(1)
df['polarization_change'] = df.bin_old != df.polarization_bin
df['extremism'] = (df.bin_old != df.polarization_bin) & ((df.polarization_bin == 'far_left')|(df.polarization_bin == 'far_right'))
df['extremism_toleft'] = (df.bin_old != df.polarization_bin) & ((df.polarization_bin == 'far_left'))
df['extremism_toright'] = (df.bin_old != df.polarization_bin) & ((df.polarization_bin == 'far_right'))
df['orient_change'] = (df.bin_old != df.polarization_bin) & (df.bin_old != 'center')
df['orient_change_toleft'] = (df.bin_old != df.polarization_bin) & (df.bin_old != 'center') & ((df.bin_old == 'center_right') |(df.bin_old == 'far_right'))
df['orient_change_toright'] = (df.bin_old != df.polarization_bin) & (df.bin_old != 'center') & ((df.bin_old == 'center_left') |(df.bin_old == 'far_left'))

# then correct for borders
mask = df.scree_name != df.scree_name.shift(1)
df['pol_old'][mask] = np.nan
df['bin_old'][mask] = np.nan
df['leaning_old'][mask] = np.nan
df['polarization_change'][mask] = np.nan
df['extremism'][mask] = np.nan
df['extremism_toleft'][mask] = np.nan
df['extremism_toright'][mask] = np.nan
df['orient_change'][mask] = np.nan
df['orient_change_toleft'][mask] = np.nan
df['orient_change_toright'][mask] = np.nan

# Convert everything to integers
df.polarization_change = df.polarization_change.replace({False: 0, True:1})
df.extremism = df.extremism.replace({False: 0, True:1})
df.extremism_toleft = df.extremism_toleft.replace({False: 0, True:1})
df.extremism_toright = df.extremism_toright.replace({False: 0, True:1})
df.orient_change = df.orient_change.replace({False: 0, True:1})
df.orient_change_toright = df.orient_change_toright.replace({False: 0, True:1})
df.orient_change_toleft = df.orient_change_toleft.replace({False: 0, True:1})

# get sentiment/emotion data from our past dataset
post_df = pd.read_csv(f"{path_to_raw}tweets_without_duplicates_regions_sentiment_demographics_topic_emotion.csv")

# Get dummies for the sentiments
post_df = pd.concat([post_df[['dates', 'sentiment', 'scree_name']], pd.get_dummies(post_df.emotion)], axis = 1)

# Get week start
post_df['dates'] = pd.to_datetime(post_df.dates)
post_df['week_start'] = post_df['dates'].dt.to_period('W').dt.start_time
post_df.drop(columns = 'dates', inplace = True)
# Get the number of tweets
post_df['n_tweets'] = 1
sum_df = post_df.groupby(['week_start', 'scree_name'], as_index=False).sum()
# Merge it with our initial dataset
df = df.merge(sum_df, how = 'left', on = ['week_start', 'scree_name'], validate = '1:1', indicator = True)
df.drop(columns = '_merge', inplace = True)

# Do the same also for the pre-covid period
completed = pd.read_pickle(f"{path_to_data}processed/pred_final.pkl.gz", compression = 'gzip')
# Get dummies for the sentiments
completed = pd.concat([completed[['dates', 'scree_name']], pd.get_dummies(completed.emotion)], axis = 1)

completed['n_tweets_old'] = 1
completed.rename(columns = {'anger': 'anger_old', 'fear': 'fear_old', 'joy': 'joy_old', 'sadness': 'sadness_old'}, inplace = True)
completed['dates'] = pd.to_datetime(completed.dates)
completed['week_start'] = completed['dates'].dt.to_period('W').dt.start_time
completed = completed[['week_start', 'scree_name', 'n_tweets_old', 'fear_old', 'anger_old', 'joy_old', 'sadness_old']]
sum_df = completed.groupby(['week_start', 'scree_name'], as_index=False).sum()
del completed
df = df.merge(sum_df, how = 'left', on = ['week_start', 'scree_name'], validate = '1:1', indicator = True)
df.drop(columns = '_merge', inplace = True)
df['n_tweets'] = np.where(pd.isnull(df.n_tweets), df['n_tweets_old'], df.n_tweets)
df.drop(columns = ['n_tweets_old'], inplace = True)
for col in ['fear', 'anger', 'joy', 'sadness']:
    df[col] = np.where(pd.isnull(df[col]), df[f'{col}_old'], df[col])
    df.drop(columns = f'{col}_old', inplace = True)

df.to_pickle(f"{path_to_processed}final_df_clean.pkl.gz", compression = 'gzip')
df.drop(columns = 'tweet_text').to_stata(f"{path_to_processed}final_df_clean.dta")

# 2. Plot Changes over time ----------------------------------------------------

# define a function that saves figures
def save_fig(fig_id, tight_layout=True):
    # The path of the figures folder ./Figures/fig_id.pdf (fig_id is a variable that you specify 
    # when you call the function)
    path = os.path.join(os.getcwd(), "Figures", fig_id + ".pdf") 
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='pdf', dpi=300)


def time_plot(dataframe, column, title, subtitle, ylabel, title_position = 0.085):

    fig, ax = plt.subplots(figsize=(15, 10))

    sns.lineplot(data = dataframe, x = 'week_start', y = column, ax = ax)
    sns.despine()

    plt.text(x= title_position, y=0.95, s= title, fontsize=30, ha="left", transform=fig.transFigure)
    plt.text(x= title_position, y=0.91, s= subtitle, fontsize=22, ha="left", transform=fig.transFigure)
    max_value = dataframe[column].max() - dataframe[column].max()*0.001
    plt.xlabel('Weeks')
    plt.ylabel(ylabel)
    pandemic = pd.datetime(2020, 2, 24)
    plt.axvline(pandemic, linewidth = 1, alpha = 1, color = 'red', linestyle = '--')
    
    annotation_dict = {
        "(1)": (2020, 3, 9)
        , '(2)': (2020, 5, 6)
        , '(3)': (2020, 6, 15)
        , '(4)': (2020, 8, 17)

    }

    for key, value in annotation_dict.items():

        week_date = dt.datetime(value[0], value[1], value[2])
        week_date = week_date - dt.timedelta(days=week_date.weekday())
        # Shift to the right the annotation
        week_date2 = dt.datetime(value[0], value[1], value[2] + 8)
        week_date2 = week_date2 - dt.timedelta(days=week_date2.weekday())

        plt.axvline(week_date,linewidth=1.5, alpha = 0.7, color='dimgrey', linestyle = '-.')
        ax.text(week_date2, max_value, key, fontsize = 20, alpha = 0.7)

    save_fig(f'{path_to_figures}{column}_overtime')

df_plot = df.loc[(df.week_start < pd.datetime(2021, 1,1))&(df.check)]
df_plot = df_plot.groupby('week_start').sum()/df_plot.groupby('week_start').count()
df_plot.reset_index(drop = False, inplace = True)
time_plot(df_plot, 'polarization_change', 'Polarization changes over time', 'Percentage of active users who changed their political alignment in a given week', 'Polarization Change (%)')
time_plot(df_plot, 'extremism', 'Extremization over time', 'Percentage of active users who became either "Far Right" or "Far Left"', 'Extremism (%)', 0.095)
time_plot(df_plot, 'orient_change', 'Orientation changes over time', 'Percentage of active users who switched from Left to Right (or viceversa)', 'Orientation Change (%)')

time_plot(df_plot, 'extremism_toright', 'Extremization over time', 'Percentage of active users who became "Far Right"', 'Extremism (%)', 0.095)
time_plot(df_plot, 'orient_change_toleft', 'Orientation changes over time', 'Percentage of active users who switched from Right to Left', 'Orientation Change (%)')

time_plot(df_plot, 'extremism_toleft', 'Extremization over time', 'Percentage of active users who became "Far Left"', 'Extremism (%)', 0.095)
time_plot(df_plot, 'orient_change_toright', 'Orientation changes over time', 'Percentage of active users who switched from Left to Right', 'Orientation Change (%)')