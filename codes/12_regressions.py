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

os.makedirs(path_to_figure_odds, exist_ok = True)

# 1. Define our main functions ------------------------------------------------------------

# Drop NAs
def do_logit(df, col, treat = 0, constant = 1, do_log = True):
    """
    Function to perform a logistic regression and give back a dataframe with the odds ratio
    df: our original dataset
    col: our dependent variable
    treat: 0 if we do want just simple event dummies, 1 if we want to compare to a treated category
    constant: 0 if we do not want to add the constant
    """
    if treat == 1:
        df_copy = df.loc[df.treat == 1]
    elif treat == 2:
        df_copy = df.loc[df.treat == 0]
    else:
        df_copy = df.copy()
    # Get dummies for each of our weeks
    df_reg = pd.concat([df_copy[col], df_copy.scree_name, pd.get_dummies(df_copy.dist), pd.get_dummies(df_copy.gender, drop_first = True), pd.get_dummies(df_copy.age, drop_first = True)], axis = 1)
    # Drop the week when COVID-19 happened (i.e. last week of February when first cases appeared)
    df_reg.drop(columns = '0 days', inplace = True)
    # Define a variable to get the distance column
    dist_col = 'dist'
    # Drop any NAs
    df_reg.dropna(axis = 0, inplace = True)
    # Set users as our index - we will then use it to cluster the standard errors
    df_reg.set_index(df_reg.scree_name, drop = True, inplace = True)
    df_reg.drop(columns = 'scree_name', inplace = True)
    # Create our Matrix of X - by adding a constant and dropping our target column
    if constant == 0:
        exog = df_reg.drop(columns = col)
    else:
        exog = sm.add_constant(df_reg.drop(columns = col))
    # Perform our logistic classification
    if do_log == True:
        logit_mod = sm.Logit(df_reg[col], exog)
    else:
        logit_mod = sm.OLS(df_reg[col], exog)
    results_logit = logit_mod.fit(cov_type = 'cluster', cov_kwds = {'groups': np.asarray(df_reg.index)})
    # Extract the parameters
    params = results_logit.params
    conf = results_logit.conf_int()
    p_value_stars = ['***' if v <= 0.001 else '**' if v <= 0.01 else '*' if v <= 0.05 else '' for v in list(results_logit.pvalues)]
    conf['Odds Ratio'] = params
    conf['P-Values'] = p_value_stars
    conf.columns = ['5%', '95%', 'Odds Ratio', 'P-Values']
    conf_odds = ['5%', '95%', 'Odds Ratio']
    # Take the exponent of everything to get the Odds-Ratios
    if do_log == True:
        for col_exp in conf_odds:
            conf[col] = np.exp(conf[col_exp])

    conf = conf[conf.index.isin(df[dist_col].unique())]
    conf.reset_index(inplace = True, drop = False)
    conf = pd.melt(conf, id_vars = ['index'], value_vars = ['5%', '95%', 'Odds Ratio'])
    conf.columns = [dist_col, 'variable', col]
    conf = conf.merge(week_list, on = dist_col, how = 'left', indicator = True, validate = 'm:1')
    assert conf._merge.value_counts()['both'] == conf.shape[0]
    return conf

# define a function that saves figures
def save_fig(fig_id, tight_layout=True):
    # The path of the figures folder ./Figures/fig_id.pdf (fig_id is a variable that you specify 
    # when you call the function)
    path = os.path.join(os.getcwd(), "Figures", fig_id + ".pdf") 
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='pdf', dpi=300)


def time_plot(dataframe, column, tag = "", y_label = "Odds Ratios"):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.ticklabel_format(style='plain')
    sns.lineplot(data = dataframe, x = 'week_start', y = column, ax = ax)
    sns.despine()

    max_value = dataframe[column].max() - dataframe[column].max()*0.001

    plt.xlabel("Week Start")
    plt.ylabel(y_label)
    plt.axvline(pandemic, linewidth = 1, alpha = 1, color = 'red', linestyle = '--')
    plt.axhline(0, linewidth = 1, alpha = 0.5, color = 'black', linestyle = '--')

    annotation_dict = {
        'DPCM #IoRestoaCasa': (2020, 3, 9)
        , 'End of Lockdown': (2020, 5, 6)
        , 'Immuni is released': (2020, 6, 15)
        , 'Nightclub closure': (2020, 8, 17)

    }

    iter_value = 0

    for key, value in annotation_dict.items():

        week_date = dt.datetime(value[0], value[1], value[2])
        week_date = week_date - dt.timedelta(days=week_date.weekday())
        title_ann = key

        plt.axvline(week_date,linewidth=1, alpha = 1, color='black')
        ax.annotate(title_ann, xy = (week_date, max_value*0.9999), xytext=(-50, 15 + iter_value), 
                        textcoords='offset points', xycoords = 'data',
                        bbox=dict(boxstyle="square", fc="white", ec="gray"), arrowprops=dict(arrowstyle='->',
                                connectionstyle="arc3"), fontsize = 18)
        iter_value += -33
        
    save_fig(f'{path_to_figure_odds}{column}{tag}')


def double_scatter(df1, df2, col1, col2, tag = ""):
    """ Function to create a scatterplot between two regression variables"""
    df1 = df1.merge(df2, on = ['dist', 'variable', 'week_start'], how = 'outer', indicator = True, validate = "1:1")
    assert df1._merge.value_counts()['both'] == df1.shape[0]
    df1 = df1.loc[df1.variable == 'Odds Ratio']
    # calculate Pearson's correlation
    pearson, _ = pearsonr(df1[col1], df1[col2])
    spear, _ = spearmanr(df1[col1], df1[col2])
    mod_ols = sm.OLS(df1[col2],df1[col1]).fit()
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.scatterplot(x=df1[col1], y=df1[col2], ax = ax)
    ax.text(df1[col1].min(), df1[col2].max(), "OLS slope: {:4.3f}, R2: {:4.3f}, Pearsons: {:4.3f}, Spearman: {:4.3f}".format(
                mod_ols.params[-1], 1-mod_ols.ssr/mod_ols.uncentered_tss, pearson, spear))
    save_fig(f'{path_to_figure_odds}{col1}_{col2}{tag}_scatter')


def double_lineplot(df1, df2, col1, col2, tag = "", ax_2 = True):
    """ 
    Function to get a seaborn plot with two time series 
    ax_2: set to True if we want to have two axis
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.lineplot(data=df1, x = 'week_start', y=col1
                , ax=ax, label = col1, legend = False)
    if ax_2 == True:
        ax2 = ax.twinx()   
        ax2.set_ylabel(col2)
    else:
        ax2 = ax
    sns.lineplot(data=df2, x = 'week_start', y=col2
                , ax=ax2, label = col2, legend = False, color = 'darkorange')
    plt.xlabel("Week Start")
    ax.set_ylabel(col1)
    plt.axvline(pandemic, linewidth = 1, alpha = 1, color = 'red', linestyle = '--')
    plt.axhline(0, linewidth = 1, alpha = 0.5, color = 'black', linestyle = '--')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc='upper right')
    save_fig(f'{path_to_figure_odds}{col1}_{col2}{tag}')

# 2. Load data and run regressions --------------------------------------------------------------


df = pd.read_pickle(f"{path_to_processed}final_df_clean.pkl.gz", compression = 'gzip')
df = df.loc[(df.week_start <= pd.datetime(2020, 12, 31))& (df.week_start > pd.datetime(2019, 12, 31))]
df.drop(columns = 'tweet_text', inplace = True)

# Get our distance from the beginning of the pandemic
pandemic = pd.datetime(2020, 2, 24)
df['dist'] = df.week_start - pandemic
df['dist'] = df.dist.astype(str)
week_list = df[['dist', 'week_start']].drop_duplicates()

# LOAD COVID-19 DATA ----------------------------------------------------------

path_to_covid = "/Users/ADORNI/Documents/COVID-19/dati-regioni/"
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
# Get the worst regions to be used as treatment
sum_df = covid.groupby('regions', as_index = False).max()
from scipy.stats import iqr
q1, q3 = np.percentile(sum_df.tot_cases, [25,75])
iqr_cases = iqr(sum_df.tot_cases)
sum_df['treat_cases'] = sum_df.tot_cases > iqr_cases
q1, q3 = np.percentile(sum_df.tot_deaths, [25,75])
iqr_deaths = iqr(sum_df.tot_deaths)
sum_df['treat_deaths'] = sum_df.tot_deaths > iqr_deaths
iqr_hosp = iqr(sum_df.ics_hosp)
sum_df['treat_hosp'] = sum_df.ics_hosp > iqr_hosp
# Plot the covid deaths/cases
cov_sum = covid.groupby('week_start', as_index = False).sum()
time_plot(cov_sum, 'tot_cases', tag = '_covid', y_label = 'COVID-19 Cases')
time_plot(cov_sum, 'tot_deaths', tag = '_covid', y_label = 'COVID-19 Deaths')


df = df.merge(covid, on =['regions', 'week_start'], how = 'left', indicator = True, validate = 'm:1')
df._merge.value_counts()

# ----------------------------------------------------------------------------

# Generate a dummy for the most affected regions
df['treat'] = df.regions.isin(sum_df.loc[sum_df.treat_deaths].regions.unique())
df['treat'].replace({False: 0, True:1}, inplace = True)
df['dist_treat'] = df.dist + "_treat"

# Get the total number of tweets as a control
df['tot_activity'] = df.groupby('week_start').n_tweets.transform('sum')
tot_tw = df.groupby('week_start', as_index = False).n_tweets.sum()
time_plot(tot_tw, 'n_tweets', y_label = 'N. of tweets')

# Create also 0/1 for being in a certain group
df = pd.concat([df, pd.get_dummies(df.polarization_bin)], axis = 1)


iter_list = ['extremism_toright', 'extremism_toleft', 'orient_change', 'orient_change_toleft',     
       'orient_change_toright', 'polarization_change', 'extremism', 'far_left', 'center_left', 'center',
       'center_right', 'far_right']

store_odds = {}
# Iterate over all our otcomes and get the odds ratio graphs
for y in iter_list:
    store_odds[y] = do_logit(df, y, treat = 0)
    time_plot(store_odds[y], y, "")
    store_odds[f"{y}_ols"] = do_logit(df, y, treat = 0, do_log = False)
    time_plot(store_odds[f"{y}_ols"], y, "_ols", y_label = y)
    store_odds[f"{y}_ols_treat"] = do_logit(df, y, treat = 1, do_log = False)
    time_plot(store_odds[f"{y}_ols_treat"], y, "_ols_treat", y_label = y)
    store_odds[f"{y}_ols_control"] = do_logit(df, y, treat = 2, do_log = False)
    time_plot(store_odds[f"{y}_ols_control"], y, "_ols_control", y_label = y)
    double_lineplot(store_odds[f"{y}_ols_treat"], store_odds[f"{y}_ols_control"], y, y, tag = "_treat_comp", ax_2 = False)

iter_emot = ['sentiment', 'anger', 'fear', 'joy', 'sadness']

for y in iter_emot:
    df[y] = df[y]/df.n_tweets
    print(y)
    conf = do_logit(df, y, treat = 0, do_log = False)
    time_plot(conf, y, "")
    store_odds[y] = conf

double_lineplot(store_odds['extremism_toright_ols'], store_odds['anger'], col1 = 'extremism_toright', col2 = 'anger', tag="_ols")
double_scatter(store_odds['extremism_toright_ols'], store_odds['anger'], col1 = 'extremism_toright', col2 = 'anger', tag="")

double_lineplot(store_odds['far_rigth_ols'], store_odds['anger'], col1 = 'far_right', col2 = 'anger', tag="_ols")
double_scatter(store_odds['far_rigth_ols'], store_odds['anger'], col1 = 'far_right', col2 = 'anger', tag="")


# Check any difference in age groups
df['treat'] = df.age.isin(['>=40', '<=18'])
df['treat'].replace({False: 0, True:1}, inplace = True)
df['dist_treat'] = df.dist + "_treat"

y = 'extremism_toright'
store_odds[f"{y}_ols_treat_age"] = do_logit(df, y, treat = 1, do_log = True)
time_plot(store_odds[f"{y}_ols_treat_age"], y, "_ols_treat_age", y_label = y)
store_odds[f"{y}_ols_control_age"] = do_logit(df, y, treat = 2, do_log = True)
time_plot(store_odds[f"{y}_ols_control_age"], y, "_ols_control_age", y_label = y)
double_lineplot(store_odds[f"{y}_ols_treat_age"], store_odds[f"{y}_ols_control_age"], y, y, tag = "_treat_comp_age", ax_2 = False)


# # Try to Fill Values
# # Create balanced panels with NaNs using stack/unstack
# df.set_index('scree_name', inplace = True, drop = False)
# df = df.set_index('week_start', append=True).unstack().stack(dropna=False)
# df['far_right']= df.groupby(level = 0).far_right.bfill().ffill()
# df['gender']= df.groupby(level = 0).gender.bfill().ffill()
# df['age']= df.groupby(level = 0).age.bfill().ffill()
# df['scree_name']= df.groupby(level = 0).scree_name.bfill().ffill()
# # Fillna() works within columns/rows; stack and unstack appropriately to use this method
# df = df.unstack('scree_name')
# df['city_code'] = df['far_right'].fillna(method='ffill').fillna(method='bfill')
# df = df.stack().swaplevel().sort_index()
# print(df)

# conf = do_logit(df, 'far_right', treat = 0, do_log = False)
# time_plot(conf, 'far_right', tag = "_prova")

# TBD - Invece che treated location, faccio Week + Week*Dosage (dosage = numero di morti per regione/settimana se trovo i dati)
# Altrimenti uso come treated le 5 locatio con piu' morti nella prima ondata/primo anno? Check also time frame

# Uso N/Tweets si o no?
# Cerco il modo di fare correlati Anger e Extremist
# Provo altre coppie?