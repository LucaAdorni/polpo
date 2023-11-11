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
import matplotlib.dates as mdates
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
path_to_tables = f"{path_to_repo}tables/"
path_to_figures_final = f"{path_to_figures}final/"

# define a function that saves figures
def save_fig(fig_id, tight_layout=True):
    # The path of the figures folder ./Figures/fig_id.png (fig_id is a variable that you specify 
    # when you call the function)
    path = os.path.join(fig_id + ".pdf") 
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='pdf', dpi=300)

# 1. Load Dataset ----------------------------------------------------

# Load our regression results 
store_odds = pd.read_stata(f"{path_to_processed}reg_results.dta")  

# with open(f'{path_to_processed}regression_res.pkl', 'rb') as f:
#     store_odds = pickle.load(f)
# Define pandemic date
pandemic = pd.datetime(2020, 2, 24)

def time_plot(dataframe, column, tag = "", y_label = "Odds Ratios", x = 'week_start'):
    # Get the correct columns to long format
    dataframe.sort_values(by = ['dist'], inplace = True)
    dataframe = pd.melt(dataframe, id_vars = [x, 'dist'], value_vars = [f'b{column}', f'ul{column}', f'll{column}'])
    # Remove the constant value
    dataframe.loc[dataframe.dist == '-7 days', 'value'] = 0
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.ticklabel_format(style='plain')
    if column == "anger":
        sns.lineplot(data = dataframe, x = x, y = "value", ax = ax, color = 'darkorange')
    else:
        sns.lineplot(data = dataframe, x = x, y = "value", ax = ax)
    sns.despine()

    if x == 'month':
        plt.xlabel("Month", fontsize = 28)
        plt.axvline(2, linewidth = 1.5, alpha = 1, color = 'red', linestyle = '--')
    elif x == 'biweek':
        plt.xlabel("Bi-weekly Start", fontsize = 28)
        plt.axvline(pandemic, linewidth = 1.5, alpha = 1, color = 'red', linestyle = '--')
        annotation_dict = {
            "1": (2020, 3, 9)
            , '2': (2020, 5, 4)
            , '3': (2020, 6, 15)
            , '4': (2020, 8, 10)
            , '5': (2020, 10, 12)
            , '6': (2020, 11, 2)
        }
    else:
        plt.xlabel("Week Start", fontsize = 35)
        plt.axvline(pandemic, linewidth = 1.5, alpha = 1, color = 'red', linestyle = '--')
        annotation_dict = {
            "1": (2020, 3, 9)
            , '2': (2020, 5, 4)
            , '3': (2020, 6, 15)
            , '4': (2020, 8, 17)
            , '5': (2020, 10, 12)
            , '6': (2020, 11, 2)
        }
    plt.ylabel(f"{y_label} (%)", fontsize = 35)
    
    plt.axhline(0, linewidth = 1, alpha = 0.3, color = 'black', linestyle = '-')
    plt.xticks(fontsize = 30)
    if column in ['far_right', 'extremism_toright']:
        plt.yticks(np.arange(-0.0025, 0.0125, 0.0025), fontsize = 30)
        max_value = 0.009
    elif column in ['center_left']:
        plt.yticks(np.arange(-0.10, 0.15, 0.05), fontsize = 30)
        max_value = 0.09
    elif column in ['center', 'center_right']:
        plt.yticks(np.arange(-0.10, 0.15, 0.05), fontsize = 30)
        max_value = 0.09
    elif column in ['anger']:
        plt.yticks(np.arange(-0.05, 0.25, 0.05),  fontsize = 30)
        max_value = 0.19
    else:
        plt.yticks(np.arange(-0.015, 0.045, 0.01), fontsize = 30)
        max_value = 0.034
    locator = mdates.AutoDateLocator(minticks=12, maxticks=14)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_formatter(formatter)
    if x == 'month':
        annotation_dict = {
            "1": 3
            , '2': 5
            , '3': 6
            , '4': 8
            , '5': 10
            , '6': 11
        }

        for key, value in annotation_dict.items():

            plt.axvline(value,linewidth=1.5, alpha = 0.7, color='dimgrey', linestyle = '-.')
            ax.text(value+0.1, max_value, key, fontsize = 28, alpha = 0.7)

    else:
        for key, value in annotation_dict.items():

            week_date = dt.datetime(value[0], value[1], value[2])
            week_date = week_date - dt.timedelta(days=week_date.weekday())
            # Shift to the right the annotation
            week_date2 = dt.datetime(value[0], value[1], value[2] + 8)
            week_date2 = week_date2 - dt.timedelta(days=week_date2.weekday())

            plt.axvline(week_date,linewidth=1.5, alpha = 0.7, color='dimgrey', linestyle = '-.')
            ax.text(week_date2, max_value, key, fontsize = 25, alpha = 0.7)
        
    save_fig(f'{path_to_figures_final}{tag}{column}')


def double_scatter(df, col1, col2, col1_name, col2_name, tag = ""):
    df1 = df.copy()
    df2 = df.copy()
    # Get the correct columns to long format
    df1.sort_values(by = ['dist'], inplace = True)
    df1 = pd.melt(df1, id_vars = ['week_start', 'dist'], value_vars = [f'b{col1}', f'ul{col1}', f'll{col1}'])
    df2.sort_values(by = ['dist'], inplace = True)
    df2 = pd.melt(df2, id_vars = ['week_start', 'dist'], value_vars = [f'b{col2}', f'ul{col2}', f'll{col2}'])
    # Remove the constant value
    df1.loc[df1.dist == '-7 days', 'value'] = 0
    # Remove the constant value
    df2.loc[df2.dist == '-7 days', 'value'] = 0
    df1.rename(columns = {'value': f"b{col1}"}, inplace = True)
    df2.rename(columns = {'value': f"b{col2}"}, inplace = True)
    """ Function to create a scatterplot between two regression variables"""
    if '_merge' in df1.columns:
        df1.drop(columns = '_merge', inplace = True)
    if '_merge' in df2.columns:
        df2.drop(columns = '_merge', inplace = True)
    df1 = df1.loc[(df1.variable == f'b{col1}')]
    df2 = df2.loc[(df2.variable == f'b{col2}')]
    df1 = df1.merge(df2, on = ['week_start'], how = 'outer', indicator = True, validate = "1:1")
    assert df1._merge.value_counts()["both"] == df1.shape[0]
    # calculate Pearson's correlation
    pearson, _ = pearsonr(df1[f"b{col1}"], df1[f"b{col2}"])
    spear, _ = spearmanr(df1[f"b{col1}"], df1[f"b{col2}"])
    mod_ols = sm.OLS(df1[f"b{col2}"],sm.add_constant(df1[f"b{col1}"])).fit()
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.regplot(x=df1[f"b{col1}"], y=df1[f"b{col2}"], ax = ax, scatter_kws = {"s": 100})
    sns.despine()
    ax.text(df1[f"b{col1}"].min(), df1[f"b{col2}"].max()*1.3,
             "OLS slope: {:4.3f} ({:4.3f})".format(
                mod_ols.params[-1], mod_ols.bse[-1]), 
                fontsize = 30)
    ax.text(df1[f"b{col1}"].min(), df1[f"b{col2}"].max()*1.15,
             "Pearson: {:4.3f}, Spearman: {:4.3f}".format(
                pearson, spear), 
                fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.xticks(fontsize = 30)
    ax.set_ylabel(f"{col2_name} (%)", fontsize = 35)
    ax.set_xlabel(f"{col1_name} (%)", fontsize = 35)
    save_fig(f'{path_to_figures_final}{tag}{col1}_{col2}_scatter')


def double_lineplot(df, col1, col2, col1_name, col2_name, label_1, label_2, tag = "", ax_2 = True):
    """ 
    Function to get a seaborn plot with two time series 
    ax_2: set to True if we want to have two axis
    """
    df1 = df.copy()
    df2 = df.copy()
    # Get the correct columns to long format
    df1.sort_values(by = ['dist'], inplace = True)
    df1 = pd.melt(df1, id_vars = ['week_start', 'dist'], value_vars = [f'b{col1}', f'ul{col1}', f'll{col1}'])
    df2.sort_values(by = ['dist'], inplace = True)
    df2 = pd.melt(df2, id_vars = ['week_start', 'dist'], value_vars = [f'b{col2}', f'ul{col2}', f'll{col2}'])
    # Remove the constant value
    df1.loc[df1.dist == '-7 days', 'value'] = 0
    # Remove the constant value
    df2.loc[df2.dist == '-7 days', 'value'] = 0
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.lineplot(data=df1, x = 'week_start', y=f"value"
                , ax=ax, label = label_1, legend = False)
    plt.yticks(fontsize = 30)
    plt.xticks(fontsize = 30)
    if ax_2 == True:
        ax2 = ax.twinx()   
        ax2.set_ylabel(f"{col2_name} (%)", fontsize = 35)
    else:
        ax2 = ax
    sns.lineplot(data=df2, x = 'week_start', y=f"value"
                , ax=ax2, label = label_2, legend = False, color = 'darkorange', linestyle = '--')
    plt.yticks(fontsize = 30)
    plt.xticks(fontsize = 30)

    if col1 in ['far_right', 'extremism_toright']:
        ax.set_yticks(np.arange(-0.0025, 0.0125, 0.0025), fontsize = 30)
    elif col1 in ['center_left']:
        ax.set_yticks(np.arange(-0.15, 0.1, 0.05), fontsize = 30)
    else:
        ax.set_yticks(np.arange(-0.05, 0.25, 0.05), fontsize = 30)
    ax2.set_yticks(np.arange(-0.05, 0.25, 0.05),  fontsize = 30)
    plt.axhline(0, linewidth = 1, alpha = 0.3, color = 'black', linestyle = '-')
    ax.set_ylabel(f"{col1_name} (%)", fontsize = 35)
    plt.axvline(pandemic, linewidth = 1, alpha = 1, color = 'red', linestyle = '--')
    # plt.axhline(0, linewidth = 1, alpha = 0.5, color = 'black', linestyle = '--')
    ax.set_xlabel("Week Start", fontsize = 35)
    locator = mdates.AutoDateLocator(minticks=12, maxticks=14)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_formatter(formatter)
    if ax_2:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax2.legend(h1+h2, l1+l2, loc='upper right', fontsize = 35, ncol = 1, frameon=False)
    else:
        plt.legend(loc = 'upper right', fontsize = 35)

    save_fig(f'{path_to_figures_final}{tag}{col1}_{col2}')


# Define a dictionary for labels
iter_dict = {
            'extremism_toright': 'Extremism to the Right',
            'extremism_toleft': "Extremism to the Left",
            'orient_change_toleft': "Orientation Change to the Left",
            'orient_change_toright': "Orientation Change to the Right",
            'far_left': "Far Left", 
            'center_left': "Center Left", 
            'center': "Center",
            'center_right': "Center Right", 
            'far_right': "Far Right"}


y = 'far_right'
emo = 'anger'
v = iter_dict[y]
double_lineplot(store_odds, col1 = y, col2 = emo, col1_name= v, col2_name = emo.capitalize(), label_1 = v, label_2 = emo.capitalize(), tag="fig_3_")
double_scatter(store_odds, col1 = y, col2 = emo, col1_name= v, col2_name = emo.capitalize(), tag="fig_3_")
        
y = 'center_right'
emo = 'anger'
v = iter_dict[y]
double_lineplot(store_odds, col1 = y, col2 = emo, col1_name= v, col2_name = emo.capitalize(), label_1 = v, label_2 = emo.capitalize(), tag="fig_3_")
double_scatter(store_odds, col1 = y, col2 = emo, col1_name= v, col2_name = emo.capitalize(), tag="fig_3_")
        
y = 'center_left'
emo = 'anger'
v = iter_dict[y]
double_lineplot(store_odds, col1 = y, col2 = emo, col1_name= v, col2_name = emo.capitalize(), label_1 = v, label_2 = emo.capitalize(), tag="fig_3_")
double_scatter(store_odds, col1 = y, col2 = emo, col1_name= v, col2_name = emo.capitalize(), tag="fig_3_")
        
# Plot anger levels
time_plot(store_odds, 'anger', "fig_a4_", y_label = 'Anger')

# APPENDIX FIGURES ---------------------------------------------------

# Iterate over all our otcomes and get the odds ratio graphs
for y, label in iter_dict.items():
    time_plot(store_odds, y, "fig_a4_", y_label = label)