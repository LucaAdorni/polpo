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
def do_logit(df, col, treat = 0, constant = 1, do_log = True, interval = 1, region_fe = False):
    """
    Function to perform a logistic regression and give back a dataframe with the odds ratio
    df: our original dataset
    col: our dependent variable
    treat: 0 if we do want just simple event dummies, 1 if we want to compare to a treated category
    constant: 0 if we do not want to add the constant
    interval: 1 for weekly regressions, 2 for bi-weekly, 4 for monthly
    """
    if interval == 4:
        dist_colname = 'dist_month'
        treat_colname = 'dist_month_treat'
        to_drop = '0'
        to_drop_treat = '0_treat'
        time_fe = 'month'
        merge_df = month_list
    elif interval == 1:
        dist_colname = 'dist'
        treat_colname = 'dist_treat'
        to_drop = '-7 days'
        to_drop_treat = '-7 days_treat'
        time_fe = 'week_start'
        merge_df = week_list
    elif interval == 2:
        dist_colname = 'dist_bi'
        treat_colname = 'dist_bi_treat'
        to_drop = '-14 days'
        to_drop_treat = '-14 days_treat'
        time_fe = 'biweek'
        merge_df = biweek_list
    if treat == 1:
        df_copy = df.loc[df.treat == 1]
    elif treat == 2:
        df_copy = df.loc[df.treat == 0]
        df_copy = df_copy.loc[pd.isnull(df_copy.regions) == False]
    else:
        df_copy = df.copy()
    # Get dummies for each of our weeks
    if treat == 3:    
        df_copy.loc[df_copy.treat == 0, treat_colname] = to_drop_treat
        df_copy = df_copy.loc[pd.isnull(df_copy.regions) == False]
        # Get the total number of tweets as a control
        df_copy['tot_activity'] = df_copy.groupby([time_fe, 'treat']).n_tweets.transform('sum')
        df_reg = pd.concat([df_copy[col], df_copy[time_fe], df_copy.tot_activity,  df_copy.scree_name, pd.get_dummies(df_copy[treat_colname]), pd.get_dummies(df_copy.treat, drop_first=True), pd.get_dummies(df_copy[dist_colname]), pd.get_dummies(df_copy.gender, drop_first = True), pd.get_dummies(df_copy.age, drop_first = True)], axis = 1)
        df_reg.drop(columns = to_drop_treat, inplace = True)
    else:
        df_reg = pd.concat([df_copy[col], df_copy[time_fe],  df_copy.tot_activity, df_copy.scree_name, pd.get_dummies(df_copy[dist_colname]), pd.get_dummies(df_copy.gender, drop_first = True), pd.get_dummies(df_copy.age, drop_first = True)], axis = 1)
    if region_fe:
        df_reg = pd.concat([df_reg, pd.get_dummies(df_copy.regions)], axis = 1)
    # Drop the week before COVID-19 happened (i.e. last week of February when first cases appeared)
    df_reg.drop(columns = to_drop, inplace = True)
    # Define a variable to get the distance column
    dist_col = dist_colname
    # Drop any NAs
    df_reg.dropna(axis = 0, inplace = True)
    # Set users as our index - we will then use it to cluster the standard errors
    df_reg.set_index([df_reg.scree_name, df_reg[time_fe]], drop = True, inplace = True)
    df_reg.drop(columns = ['scree_name', time_fe], inplace = True)
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
    if treat == 3:
        conf = conf[conf.index.isin(df[treat_colname].unique())]
    else:
        conf = conf[conf.index.isin(df[dist_colname].unique())]
    conf.reset_index(inplace = True, drop = False)
    conf = pd.melt(conf, id_vars = ['index'], value_vars = ['5%', '95%', 'Odds Ratio'])
    conf.columns = [dist_colname, 'variable', col]
    if treat == 3:
        conf[dist_colname] = conf[dist_colname].apply(lambda x: re.sub('_treat', "", x))
    conf = conf.merge(merge_df, on = dist_colname, how = 'left', indicator = True, validate = 'm:1')
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

        }
    else:
        plt.xlabel("Week Start", fontsize = 28)
        plt.axvline(pandemic, linewidth = 1.5, alpha = 1, color = 'red', linestyle = '--')
        annotation_dict = {
            "(1)": (2020, 3, 9)
            , '(2)': (2020, 5, 4)
            , '(3)': (2020, 6, 15)
            , '(4)': (2020, 8, 17)

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
        
    save_fig(f'{path_to_figure_odds}{column}{tag}')


def double_scatter(df1, df2, col1, col2, col1_name, col2_name, tag = ""):
    """ Function to create a scatterplot between two regression variables"""
    if '_merge' in df1.columns:
        df1.drop(columns = '_merge', inplace = True)
    if '_merge' in df2.columns:
        df2.drop(columns = '_merge', inplace = True)
    df1 = df1.merge(df2, on = ['dist', 'variable', 'week_start'], how = 'outer', indicator = True, validate = "1:1")
    assert df1._merge.value_counts()['both'] == df1.shape[0]
    df1 = df1.loc[df1.variable == 'Odds Ratio']
    # calculate Pearson's correlation
    pearson, _ = pearsonr(df1[col1], df1[col2])
    spear, _ = spearmanr(df1[col1], df1[col2])
    mod_ols = sm.OLS(df1[col2],df1[col1]).fit()
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.scatterplot(x=df1[col1], y=df1[col2], ax = ax, s = 100)
    ax.text(df1[col1].min(), df1[col2].max(), "OLS slope: {:4.3f}, R2: {:4.3f}, Pearsons: {:4.3f}, Spearman: {:4.3f}".format(
                mod_ols.params[-1], 1-mod_ols.ssr/mod_ols.uncentered_tss, pearson, spear), fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.xticks(fontsize = 20)
    ax.set_ylabel(col2_name, fontsize = 28)
    ax.set_xlabel(col1_name, fontsize = 28)
    save_fig(f'{path_to_figure_odds}{col1}_{col2}{tag}_scatter')


def double_lineplot(df1, df2, col1, col2, col1_name, col2_name, label_1, label_2, tag = "", ax_2 = True):
    """ 
    Function to get a seaborn plot with two time series 
    ax_2: set to True if we want to have two axis
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.lineplot(data=df1, x = 'week_start', y=col1
                , ax=ax, label = label_1, legend = False)
    plt.yticks(fontsize = 20)
    plt.xticks(fontsize = 20)
    if ax_2 == True:
        ax2 = ax.twinx()   
        ax2.set_ylabel(col2_name, fontsize = 28)
    else:
        ax2 = ax
    sns.lineplot(data=df2, x = 'week_start', y=col2
                , ax=ax2, label = label_2, legend = False, color = 'darkorange')
    plt.yticks(fontsize = 20)
    plt.xticks(fontsize = 20)
    ax.set_ylabel(col1_name, fontsize = 28)
    plt.axvline(pandemic, linewidth = 1, alpha = 1, color = 'red', linestyle = '--')
    # plt.axhline(0, linewidth = 1, alpha = 0.5, color = 'black', linestyle = '--')
    ax.set_xlabel("Week Start", fontsize = 28)
    if ax_2:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax2.legend(h1+h2, l1+l2, loc='upper right', fontsize = 28)
    else:
        plt.legend(loc = 'upper right', fontsize = 28)
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

path_to_covid = "/Users/ADORNI/Documents/COVID-19/dati-regioni/"
path_to_covid = "/Users/luca9/Documenti/COVID-19/dati-regioni/"
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

# 3. Run the Regressions

iter_dict = {'extremism_toright': 'Extremism to the Right',
             'extremism_toleft': 'Extremism to the Left', 
             'orient_change': "Orientation change", 
             'orient_change_toleft': "Orientation change to the Left",     
            'orient_change_toright': "Orientation change to the Right", 
            'polarization_change': "Polarization change", 
            'extremism': "Extremism", 
            'far_left': "Far Left", 
            'center_left': "Center Left", 
            'center': "Center",
            'center_right': "Center Right", 
            'far_right': "Far Right"}

store_odds = {}
# Iterate over all our otcomes and get the odds ratio graphs
for y, label in iter_dict.items():
    store_odds[f"{y}_ols"] = do_logit(df, y, treat = 0, do_log = False)
    time_plot(store_odds[f"{y}_ols"], y, "_ols", y_label = label)
    store_odds[f"{y}_ols_did"] = do_logit(df, y, treat = 3, do_log = False, region_fe=True)
    time_plot(store_odds[f"{y}_ols_did"], y, "_ols_did", y_label = f"{label} (DiD)")
    store_odds[f"{y}_ols_did_b"] = do_logit(df, y, treat = 3, do_log = False, interval = 2, region_fe=True)
    time_plot(store_odds[f"{y}_ols_did_b"], y, "_ols_did_biweek", y_label = f"{label} (DiD)", x = 'biweek')
    store_odds[f"{y}_ols_did_m"] = do_logit(df, y, treat = 3, do_log = False, interval = 4, region_fe=True)
    time_plot(store_odds[f"{y}_ols_did_m"], y, "_ols_did_month", y_label = f"{label} (DiD)", x = 'month')

iter_emot = ['sentiment', 'anger', 'fear', 'joy', 'sadness']

for y in iter_emot:
    df[y] = df[y]/df.n_tweets
    print(y)
    conf = do_logit(df, y, treat = 0, do_log = False)
    time_plot(conf, y, "", y_label = y.capitalize())
    store_odds[y] = conf
    conf = do_logit(df, y, treat = 3, do_log = False)
    time_plot(conf, y, "_did", y_label = f"{y.capitalize()} (DiD)")
    store_odds[f"{y}_did"] = conf

for y, v in iter_dict.items():
    for emo in iter_emot:
        double_lineplot(store_odds[f'{y}_ols'], store_odds[emo], col1 = y, col2 = emo, col1_name= v, col2_name = emo.capitalize(), label_1 = v, label_2 = emo.capitalize(), tag="_ols")
        double_scatter(store_odds[f'{y}_ols'], store_odds[emo], col1 = y, col2 = emo, col1_name= v, col2_name = emo.capitalize(), tag="")
        double_lineplot(store_odds[f'{y}_ols_did'], store_odds[f"{emo}_did"], col1 = y, col2 = emo, col1_name= f"{v} (DiD)", col2_name = f"{emo.capitalize()} (DiD)", label_1 = v, label_2 = emo.capitalize(), tag="_ols_did")
        double_scatter(store_odds[f'{y}_ols_did'], store_odds[f"{emo}_did"], col1 = y, col2 = emo, col1_name= f"{v} (DiD)", col2_name = f"{emo.capitalize()} (DiD)", tag="_did")


# Get the total number of tweets as a control
df['tot_activity'] = df.groupby(['week_start', 'treat']).n_tweets.transform('sum')
df['users'] = 1
tot_tw = df.groupby(['week_start', 'treat'], as_index = False)['users','tot_activity'].sum()

double_lineplot(tot_tw.loc[tot_tw.treat == 1], tot_tw.loc[tot_tw.treat == 0], col1 = 'users', col2 = 'users', col1_name= 'Active Users', col2_name = 'Active Users', label_1 = "Treatment", label_2 = "Control", tag="_treat_comp", ax_2 = False)
double_lineplot(tot_tw.loc[tot_tw.treat == 1], tot_tw.loc[tot_tw.treat == 0], col1 = 'tot_activity', col2 = 'tot_activity', col1_name= 'Tweets', col2_name = 'Tweets', label_1 = "Treatment", label_2 = "Control", tag="_treat_comp", ax_2 = False)


# Political Survey Comparison ------------------------------------

# we import survey data on political alignment
survey = pd.read_excel(f'{path_to_data}political_survey.xlsx')

survey["date"] = pd.to_datetime(survey.Partito) # create a date variable
survey["month"] = survey.date.dt.month # extract the month
survey = survey[survey.date.dt.year == 2020] # filter all non 2020 dates
survey = survey[survey.month <= 9]

melted_survey = pd.melt(survey, id_vars = ['Partito','date', 'month'])
melted_survey.rename(columns = {'variable': 'Party'}, inplace = True)
melted_survey = melted_survey[(melted_survey.Party != 'Coraggio Italia')&(melted_survey.Party != 'Cambiamo')&(melted_survey.Party != 'SX')&(melted_survey.Party != 'Europa Verde')&(melted_survey.Party != 'Cambiamo!')]
# melted_survey = melted_survey[(melted_survey.Party != 'SX')&(melted_survey.Party != 'Cambiamo!')&(melted_survey.Party != 'Coraggio Italia')&(melted_survey.Party != 'Europa Verde')&(melted_survey.Party != 'Più Europa')&(melted_survey.Party != 'Italia Viva')&(melted_survey.Party != 'Azione')]


# CHECK DID-------------------------------------------

df['treat'] = df.regions.isin(sum_df.loc[sum_df.treat_cases].regions.unique())
df['treat'].replace({False: 0, True:1}, inplace = True)
df['dist_treat'] = df.dist + "_treat"
df['dist_month_treat'] = df.dist_month + "_treat"
y = 'extremism'
label = 'Extremism'
store_odds[f"{y}_ols_did_2"] = do_logit(df, y, treat = 3, do_log = False)
time_plot(store_odds[f"{y}_ols_did_2"], y, "_ols_did_2", y_label = f"{label} (DiD)")

store_odds[f"{y}_ols_control"] = do_logit(df, y, treat = 2, do_log = False)
store_odds[f"{y}_ols_treat"] = do_logit(df, y, treat = 1, do_log = False)

double_lineplot(store_odds[f'{y}_ols_treat'], store_odds[f"{y}_ols_control"], col1 = y, col2 = y, col1_name=y, col2_name=y, label_1= f"Treat", label_2 = f"Control", tag="_did_comp", ax_2 = False, )


# Get the sum over months/regions
df['user'] = 1
regions = df.loc[df.regions != np.nan].groupby(['week_start', 'regions', 'dist'], as_index = False).sum()

for col in iter_dict.keys():
    regions[f'{col}_p'] = regions[col]/regions['user']



regions.head()


sns.lineplot(data = regions, x = 'week_start', y = 'extremism_p', hue = 'regions')

plt.show()

nord = ['Abruzzo', 'Basilicata', 'Calabria', 'Campania', 'Emilia-Romagna', 
'Friuli-Venezia Giulia', 'Lazio', 'Liguria', 'Lombardia', 'Marche', 
'Molise', 'Piemonte', 'Puglia', 'Sardegna', 'Sicilia', 'Toscana', 
'Trentino-Alto Adige', 'Umbria', "Valle d'Aosta", 'Veneto']


nord = ['Emilia-Romagna', 
'Friuli-Venezia Giulia', 'Liguria', 'Lombardia', 'Piemonte', 'Toscana', 
'Trentino-Alto Adige', "Valle d'Aosta", 'Veneto']


nord = ['Emilia-Romagna', 'Lazio', 'Lombardia', 'Piemonte', 'Sicilia', 'Toscana', 'Veneto']

nord = ['Emilia-Romagna', 'Lazio', 'Lombardia', 'Piemonte', 'Toscana', 'Veneto']

['Campania', 'Emilia-Romagna', 'Lazio', 'Lombardia', 'Piemonte', 'Veneto']

nord = ['Emilia-Romagna', 'Lazio', 'Lombardia', 'Piemonte', 'Sicilia', 'Toscana', 'Veneto', 'Campania']

regions['nord'] = regions.regions.isin(nord)


regions['treat'] = regions.regions.isin(nord)
regions['treat'].replace({False: 0, True:1}, inplace = True)
regions['dist_treat'] = regions.dist + "_treat"

df['nord'] = df.regions.isin(['Emilia-Romagna', 
'Friuli-Venezia Giulia', 'Liguria', 'Lombardia', 'Piemonte',
'Trentino-Alto Adige', "Valle d'Aosta", 'Veneto'])

df['centro'] = df.regions.isin(['Toscana', 'Marche', 'Umbria', 'Lazio'])

df['sud'] = df.regions.isin(['Abruzzo', 'Basilicata', 'Calabria', 'Campania', 'Puglia', 'Molise', 'Sicilia', 'Sardegna'])


df_copy = df.copy()

df_copy['dist_treat_n'] = df.dist + "_treat_n"
df_copy['dist_treat_c'] = df.dist + "_treat_c"
df_copy['dist_treat_s'] = df.dist + "_treat_s"
df_copy['nord'].replace({False: 0, True:1}, inplace = True)
df_copy['sud'].replace({False: 0, True:1}, inplace = True)
df_copy['centro'].replace({False: 0, True:1}, inplace = True)
  

df_copy['reg_c'] = np.where(df_copy.nord == 1, 1, np.where(df_copy.sud == 1, 2, 0))

df_copy.loc[df_copy.nord == 0, 'dist_treat_n'] = '-7 days_treat_n'
df_copy.loc[df_copy.sud == 0, 'dist_treat_s'] = '-7 days_treat_s'
df_copy.loc[df_copy.centro == 0, 'dist_treat_c'] = '-7 days_treat_c'
df_copy = df_copy.loc[pd.isnull(df_copy.regions) == False]
# Get the total number of tweets as a control
df_copy['tot_activity'] = df_copy.groupby(['week_start', 'reg_c']).n_tweets.transform('sum')
df_reg = pd.concat([df_copy.regions,df_copy['orient_change_toleft'], df_copy.tot_activity, df_copy.week_start, pd.get_dummies(df_copy.dist_treat_n), pd.get_dummies(df_copy.dist_treat_s), pd.get_dummies(df_copy.reg_c, drop_first=True), pd.get_dummies(df_copy.dist)], axis = 1)
df_reg.drop(columns = ['-7 days_treat_n', '-7 days_treat_s'], inplace = True)
# Drop the week before COVID-19 happened (i.e. last week of February when first cases appeared)
df_reg.drop(columns = '-7 days', inplace = True)
# Define a variable to get the distance column
dist_col = 'dist'
# Drop any NAs
df_reg.dropna(axis = 0, inplace = True)
# Set users as our index - we will then use it to cluster the standard errors
df_reg.set_index([df_reg.regions, df_reg.week_start], drop = True, inplace = True)
df_reg.drop(columns = ['regions', 'week_start'], inplace = True)
# Create our Matrix of X - by adding a constant and dropping our target column

exog = sm.add_constant(df_reg.drop(columns = 'orient_change_toleft'))
# Perform our logistic classification

logit_mod = sm.OLS(df_reg['orient_change_toleft'], exog)
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
conf = conf[(conf.index.isin(df_copy['dist_treat_s'].unique()))|(conf.index.isin(df_copy['dist_treat_n'].unique()))]
conf.reset_index(inplace = True, drop = False)
conf = pd.melt(conf, id_vars = ['index'], value_vars = ['5%', '95%', 'Odds Ratio'])
conf.columns = [dist_col, 'variable', 'extremism_toright']
conf['dist_n'] = conf.dist.apply(lambda x: re.sub('_treat_n', "", x)).isin(df.dist.unique())
conf['dist'] = conf.dist.apply(lambda x: re.sub('_treat_(n|s)', "", x))
conf = conf.merge(week_list, on = dist_col, how = 'left', indicator = True, validate = 'm:1')
assert conf._merge.value_counts()['both'] == conf.shape[0]
sns.lineplot(data = conf, x = 'week_start', y = 'extremism_toright', hue = 'dist_n')
plt.axhline(0)
plt.show()

sns.lineplot(data = regions, x = 'week_start', y = 'extremism_toright_p', hue = 'nord')

plt.show()

prova = regions.groupby(['week_start', 'dist', 'nord'], as_index = False).sum()
for col in iter_dict.keys():
    prova[f'{col}_p'] = prova[col]/prova['user']


sns.lineplot(data = prova, x = 'week_start', y = 'extremism_toright_p', hue = 'nord')

plt.show()


store_odds = {}
df['treat'] = df.regions.isin(nord)
df['treat'].replace({False: 0, True:1}, inplace = True)
df['dist_treat'] = df.dist + "_treat"
y = 'extremism_toright'
label = 'Extremism to the Right'
df_fix = df.copy()
df_fix['regions'] = df_fix['regions'].fillna("")
store_odds[f"{y}_ols_did_2"] = do_logit(df_fix, y, treat = 3, do_log = False)
time_plot(store_odds[f"{y}_ols_did_2"], y, "_ols_did_3", y_label = f"{label} (DiD)")


prova = do_logit(df, 'far_left', treat = 3, do_log = False, do_monthly=True)

sns.lineplot(data = prova, y = 'far_left', x = 'month')
plt.axhline(0, color = 'b')
plt.show()

time_plot(store_odds[f"{y}_ols_did_2"], y, "_ols_did_2", y_label = f"{label} (DiD)")




# Generate a dummy for the most affected regions
df['treat'] = df.regions.isin(["Lombardia", "Veneto", "Campania"])
df['treat'].replace({False: 0, True:1}, inplace = True)
df['dist_treat'] = df.dist + "_treat"
df['dist_month_treat'] = df.dist_month + "_treat"
df['dist_bi_treat'] = df.dist_bi + "_treat"

store_focus = {}
# Iterate over all our otcomes and get the odds ratio graphs
for y, label in iter_dict.items():
    store_focus[f"{y}_ols_did"] = do_logit(df, y, treat = 3, do_log = False)
    time_plot(store_focus[f"{y}_ols_did"], y, "_ols_did_focus", y_label = f"{label} (DiD)")
    store_focus[f"{y}_ols_did_b"] = do_logit(df, y, treat = 3, do_log = False, interval = 2)
    time_plot(store_focus[f"{y}_ols_did_b"], y, "_ols_did_biweek_focus", y_label = f"{label} (DiD)", x = 'biweek')
    store_focus[f"{y}_ols_did_m"] = do_logit(df, y, treat = 3, do_log = False, interval = 4)
    time_plot(store_focus[f"{y}_ols_did_m"], y, "_ols_did_month_focus", y_label = f"{label} (DiD)", x = 'month')



# COMPARE SINGLE MOST-AFFECTED REGIONS TO CONTROLS
controls = sum_df.loc[(sum_df.treat_cases == False) & (sum_df.treat_deaths == False)].regions.unique().tolist()

for tr_region in ["Lombardia", "Veneto", "Campania", "Piemonte", "Valle d'Aosta", "Emilia-Romagna"]:
        
    # Generate a dummy for the most affected regions
    df['treat'] = df.regions.isin([tr_region])
    df['treat'].replace({False: 0, True:1}, inplace = True)
    df['dist_treat'] = df.dist + "_treat"
    df['dist_month_treat'] = df.dist_month + "_treat"
    df['dist_bi_treat'] = df.dist_bi + "_treat"

    # Drop treated regions
    df_r = df.loc[df.regions.isin(controls + [tr_region])]

    iter_dict = {'extremism_toright': 'Extremism to the Right',
                'extremism_toleft': 'Extremism to the Left', 
                'orient_change': "Orientation change", 
                'orient_change_toleft': "Orientation change to the Left",     
                'orient_change_toright': "Orientation change to the Right", 
                'polarization_change': "Polarization change", 
                'extremism': "Extremism", 
                'far_left': "Far Left", 
                'center_left': "Center Left", 
                'center': "Center",
                'center_right': "Center Right", 
                'far_right': "Far Right"}
    
    if tr_region == "Valle d'Aosta":
        tr_region = "valle_aosta"
    else:
        tr_region = tr_region.lower()

    store_odds = {}
    # Iterate over all our otcomes and get the odds ratio graphs
    for y, label in iter_dict.items():
        store_odds[f"{y}_ols_did"] = do_logit(df_r, y, treat = 3, do_log = False, region_fe=True)
        time_plot(store_odds[f"{y}_ols_did"], y, f"_ols_did{tr_region}", y_label = f"{label} (DiD)")
        store_odds[f"{y}_ols_did_b"] = do_logit(df_r, y, treat = 3, do_log = False, interval = 2, region_fe=True)
        time_plot(store_odds[f"{y}_ols_did_b"], y, f"_ols_did_biweek{tr_region}", y_label = f"{label} (DiD)", x = 'biweek')
        store_odds[f"{y}_ols_did_m"] = do_logit(df_r, y, treat = 3, do_log = False, interval = 4, region_fe=True)
        time_plot(store_odds[f"{y}_ols_did_m"], y, f"_ols_did_month{tr_region}", y_label = f"{label} (DiD)", x = 'month')
