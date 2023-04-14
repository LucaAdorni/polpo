###########################################################
# Code to create the dataset for analysis
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

# Specify here where the COVID-19 data has been downloaded
path_to_covid = "/Users/ADORNI/Documents/COVID-19/dati-regioni/"


# 1. Define our main functions ------------------------------------------------------------

# define a function that saves figures
def save_fig(fig_id, tight_layout=True):
    # The path of the figures folder ./Figures/fig_id.pdf (fig_id is a variable that you specify 
    # when you call the function)
    path = os.path.join(fig_id + ".pdf") 
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='pdf', dpi=300)

def time_plot(dataframe, column, tag = "", y_label = "Odds Ratios", x = 'week_start'):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.ticklabel_format(style='plain')
    sns.lineplot(data = dataframe, x = x, y = column, ax = ax)
    sns.despine()

    max_value = dataframe[column].max() - dataframe[column].max()*0.001
    if x == 'month':
        plt.xlabel("Month", fontsize = 28)
        plt.axvline(2, linewidth = 1.5, alpha = 1, color = 'red', linestyle = '--')
    elif x == 'biweek':
        plt.xlabel("Bi-weekly Start", fontsize = 28)
        plt.axvline(pandemic, linewidth = 1.5, alpha = 1, color = 'red', linestyle = '--')
        annotation_dict = {
            "(1)": (2020, 3, 9)
            , '(2)': (2020, 5, 4)
            , '(3)': (2020, 6, 15)
            , '(4)': (2020, 8, 10)
            , '(5)': (2020, 10, 12)
            , '(6)': (2020, 11, 2)
        }
    else:
        plt.xlabel("Week Start", fontsize = 28)
        plt.axvline(pandemic, linewidth = 1.5, alpha = 1, color = 'red', linestyle = '--')
        annotation_dict = {
            "(1)": (2020, 3, 9)
            , '(2)': (2020, 5, 4)
            , '(3)': (2020, 6, 15)
            , '(4)': (2020, 8, 17)
            , '(5)': (2020, 10, 12)
            , '(6)': (2020, 11, 2)
        }
    plt.ylabel(y_label, fontsize = 28)
    
    plt.axhline(0, linewidth = 1, alpha = 0.5, color = 'black', linestyle = '--')
    plt.yticks(fontsize = 20)
    plt.xticks(fontsize = 20)

    if x == 'month':
        annotation_dict = {
            "(1)": 3
            , '(2)': 5
            , '(3)': 6
            , '(4)': 8
            , '(5)': 10
            , '(6)': 11
        }

        for key, value in annotation_dict.items():

            plt.axvline(value,linewidth=1.5, alpha = 0.7, color='dimgrey', linestyle = '-.')
            ax.text(value+0.1, max_value, key, fontsize = 20, alpha = 0.7)

    else:
        for key, value in annotation_dict.items():

            week_date = dt.datetime(value[0], value[1], value[2])
            week_date = week_date - dt.timedelta(days=week_date.weekday())
            # Shift to the right the annotation
            week_date2 = dt.datetime(value[0], value[1], value[2] + 8)
            week_date2 = week_date2 - dt.timedelta(days=week_date2.weekday())

            plt.axvline(week_date,linewidth=1.5, alpha = 0.7, color='dimgrey', linestyle = '-.')
            ax.text(week_date2, max_value, key, fontsize = 20, alpha = 0.7)
        
    save_fig(f'{path_to_figures}{column}{tag}')

# 2. Load Dataset ----------------------------------------------------


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


# 3. ADD COVID-19 DATA ------------------------------------------------------------

df = pd.read_pickle(f"{path_to_processed}final_df_clean.pkl.gz", compression = 'gzip')
df = df.loc[(df.week_start <= pd.datetime(2020, 12, 31))& (df.week_start > pd.datetime(2019, 12, 31))]
df.drop(columns = 'tweet_text', inplace = True)

# Get our distance from the beginning of the pandemic
pandemic = pd.datetime(2020, 2, 24)
df['dist'] = df.week_start - pandemic
df['dist'] = df.dist.astype(str)
week_list = df[['dist', 'week_start']].drop_duplicates()

# Get biweekly frequencies
frequency_2 = '2W-MON'
bi_weeks = df.groupby([pd.Grouper(key="week_start", freq=frequency_2)]).week_start.agg(['min', 'max'])
bi_weeks.reset_index(inplace = True, drop = True)
bi_weeks.columns = ['lb', 'ub']
bi_weeks['bi_week'] = bi_weeks['lb']
# Merge it back
df = df.merge(bi_weeks[['bi_week','lb']], left_on = 'week_start', right_on = 'lb', validate = 'm:1', how = 'left')
df = df.merge(bi_weeks[['bi_week','ub']], left_on = 'week_start', right_on = 'ub', validate = 'm:1', how = 'left', suffixes = ("","_ub"))
df['biweek'] = df['bi_week'].fillna(df.bi_week_ub)
assert df.biweek.isna().any() == False
df.drop(columns = ['ub', 'lb', 'bi_week', 'bi_week_ub'], inplace = True)

df['dist_bi'] = df.biweek - pandemic
df['dist_bi'] = df.dist_bi.astype(str)
biweek_list = df[['dist_bi', 'biweek']].drop_duplicates()

# Get also the months
df['month'] = df.week_start.dt.month
df['dist_month'] = df.month - 2
df['dist_month'] = df['dist_month'].astype(str)
month_list = df[['dist_month', 'month']].drop_duplicates()

# LOAD COVID-19 DATA ----------------------------------------------------------

covid = []
for file in os.listdir(path_to_covid):
    f = pd.read_csv(f"{path_to_covid}{file}")
    covid.append(f)
covid = pd.concat(covid)
# Get the first week data
covid['week_start'] = pd.to_datetime(covid.data)
covid['week_start'] = covid['week_start'].dt.to_period('W').dt.start_time
covid.drop(columns = ['data', 'stato', 'codice_regione', 'lat', 'long'], inplace = True)
covid.denominazione_regione.replace({'Friuli Venezia Giulia': 'Friuli-Venezia Giulia', 'P.A. Trento': 'Trentino-Alto Adige', 'P.A. Bolzano': 'Trentino-Alto Adige'}, inplace = True)
covid.rename(columns = {'denominazione_regione': 'regions'}, inplace = True)
# Get the weekly totals
covid['tot_cases'] = covid.groupby(['regions', 'week_start']).totale_positivi.transform('max')
covid['tot_deaths'] = covid.groupby(['regions', 'week_start']).deceduti.transform('max')
covid['new_cases'] = covid.groupby(['regions', 'week_start']).nuovi_positivi.transform('sum')
covid['ics_hosp'] = covid.groupby(['regions', 'week_start']).terapia_intensiva.transform('mean')
# Drop everything we do not need
covid = covid[['week_start','regions', 'tot_cases', 'tot_deaths', 'new_cases', 'ics_hosp']]
covid.drop_duplicates(inplace = True)
# Subset to the period considered
covid = covid.loc[(covid.week_start <= pd.datetime(2020, 12, 31))]
# Get the worst regions to be used as treatment -----------------------
sum_df = covid.groupby('regions', as_index = False).max()
pop = pd.read_csv(f"{path_to_data}popolazione_regioni.csv")
sum_df = sum_df.merge(pop, on = "regions", how = "left", validate = "1:1")
# Get the % of affected
for col in ["tot_cases", "tot_deaths", "ics_hosp"]:
    sum_df[col] = sum_df[col]/sum_df["pop"]
from scipy.stats import iqr
q1, q3 = np.percentile(sum_df.tot_cases, [25,75])
iqr_cases = iqr(sum_df.tot_cases)
sum_df['treat_cases'] = sum_df.tot_cases > iqr_cases + q1
q1, q3 = np.percentile(sum_df.tot_deaths, [25,75])
iqr_deaths = iqr(sum_df.tot_deaths)
sum_df['treat_deaths'] = sum_df.tot_deaths > iqr_deaths + q1
q1, q3 = np.percentile(sum_df.ics_hosp, [25,75])
iqr_hosp = iqr(sum_df.ics_hosp)
sum_df['treat_hosp'] = sum_df.ics_hosp > iqr_hosp + q1
# Build a composite measure
sum_df["composite"] = sum_df.tot_cases * sum_df.tot_deaths
q1, q3 = np.percentile(sum_df.composite, [25,75])
iqr_composite = iqr(sum_df.composite)
sum_df['treat_composite'] = sum_df.composite > iqr_composite + q1
# Plot the covid deaths/cases -----------------------------------------
cov_sum = covid.groupby('week_start', as_index = False).sum()
time_plot(cov_sum, 'tot_cases', tag = '_covid', y_label = 'COVID-19 Cases')
time_plot(cov_sum, 'tot_deaths', tag = '_covid', y_label = 'COVID-19 Deaths')

# Merge it with our initial dataset
df = df.merge(covid, on =['regions', 'week_start'], how = 'left', indicator = True, validate = 'm:1')
df._merge.value_counts()

# ----------------------------------------------------------------------------

# Generate a dummy for the most affected regions
affected = sum_df.loc[(sum_df.treat_hosp)].regions.unique().tolist()
affected = sum_df.loc[sum_df.treat_composite].regions.unique().tolist()
df['treat'] = df.regions.isin(affected)
df['treat'].replace({False: 0, True:1}, inplace = True)
df['dist_treat'] = df.dist + "_treat"
df['dist_month_treat'] = df.dist_month + "_treat"
df['dist_bi_treat'] = df.dist_bi + "_treat"

# Get the total number of tweets as a control
df['tot_activity'] = df.groupby('week_start').n_tweets.transform('sum')
tot_tw = df.groupby('week_start', as_index = False).n_tweets.sum()
time_plot(tot_tw, 'n_tweets', y_label = 'N. of tweets')

# Create also 0/1 for being in a certain group
df = pd.concat([df, pd.get_dummies(df.polarization_bin)], axis = 1)

# Save it
df.to_pickle(f"{path_to_processed}final_df_analysis.pkl.gz", compression = 'gzip')

# Produce an export for stata
df[['scree_name', 'orient_change_toleft', 'orient_change_toright', 'extremism_toleft', 'extremism_toright', 'center', 'center_left', 'center_right', 'far_left', 'far_right'
    , 'sentiment', 'anger', 'fear', 'joy', 'sadness', 'n_tweets', 'tot_activity', 'gender', 
    'age', 'week_start', 'dist', 'regions', 'treat']].to_stata(f"{path_to_processed}final_df_analysis.dta.gz", compression = 'gzip'
                                           , convert_dates= {'week_start': '%tw'},
                                           version = 117)