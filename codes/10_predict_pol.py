###########################################################
# Code to predict polarization
# Author: Luca Adorni
# Date: January 2023
###########################################################

# 0. Setup -------------------------------------------------

import numpy as np
import pandas as pd
import os
import sys
import re
import pickle
import random
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import dill
import re
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, random_split
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Callable, Dict, Generator, List, Tuple, Union
from pathlib import PurePosixPath
import itertools
from torch import nn
from tqdm import tqdm

try:
    # Setup Repository
    with open("repo_info.txt", "r") as repo_info:
        path_to_repo = repo_info.readline()
except:
    path_to_repo = f"{os.getcwd()}/polpo/"
    sys.path.append(f"{os.getcwd()}/.local/bin") # import the temporary path where the server installed the module


print(path_to_repo)

path_to_data = f"{path_to_repo}data/"
path_to_figures = f"{path_to_repo}figures/"
path_to_tables = f"{path_to_repo}tables/"
path_to_raw = f"{path_to_data}raw/"
path_to_links = f"{path_to_data}links/"
path_to_topic = f"{path_to_data}topic/"
path_to_results = f"{path_to_data}results/"
path_to_models = f"{path_to_data}models/"
path_to_models_pol = f"{path_to_models}ML/"
path_to_processed = f"{path_to_data}processed/"


# PARAMETERS ---------------
random_seed = 42
random.seed(random_seed)

bert_tag = '_256'
percentage_filter = 0.4
learning_rate = 3e-05
batch_size = 32
epoch_num = 0
chunksize = 256


merge_tweets = True # set to True if we want to merge all tweets
only_politics = False # set to True if we want to keep only politic tweets

if only_politics:
    politics_tag = '_politics'
else:
    politics_tag = ''

if merge_tweets:
    merged_tag = '_merged'
else:
    merged_tag = '_individual'

# ----------------------

# 1. Import our dataset -------------------------------------------------

try:
    # Load our dataset, if we have already processed it
    pred = pd.read_pickle(f"{path_to_processed}df_processed.pkl.gz", compression = 'gzip')
    print("Dataset Loaded")
    print(pred.shape[0])
except:
    # PRE COVID ----------------------
    pre_df = pd.read_pickle(f"{path_to_data}processed/pred_final.pkl.gz", compression = 'gzip')

    # For the pre-covid data, change the topic column
    pre_df.loc[pre_df.final_pred == 1, 'topic'] = 'economics'
    pre_df.loc[pre_df.final_pred == 2, 'topic'] = 'politics'

    pre_df = pre_df.loc[(pre_df.topic == 'politics') | (pre_df.topic == 'economics')]

    # Get the beginning of weeks
    pre_df.dates = pd.to_datetime(pre_df.dates)
    pre_df['week_start'] = pre_df['dates'].dt.to_period('W').dt.start_time
    # And remove unnecessary columns
    pre_df = pre_df[['scree_name', 'tweet_text', 'week_start']]

    # POST COVID ----------------------
    post_df = pd.read_csv(f"{path_to_raw}tweets_without_duplicates_regions_sentiment_demographics_topic_emotion.csv")

    post_df = post_df.loc[post_df.is_org == 'non-org']

    # we limit our training set to tweets talking about politics
    post_df = post_df.loc[(post_df.topic == 'politics') | (post_df.topic == 'economics')]

    # Get the beginning of weeks
    post_df.dates = pd.to_datetime(post_df.dates)
    post_df['week_start'] = post_df['dates'].dt.to_period('W').dt.start_time
    # And remove unnecessary columns
    post_df = post_df[['scree_name', 'tweet_text', 'week_start']]

    # Merge the tweets
    post_df = pd.concat([post_df, pre_df])
    pred = post_df.groupby(['scree_name', 'week_start']).tweet_text.apply(' '.join).reset_index()
    
    # Drop any duplicates
    pred.drop_duplicates(subset = ['scree_name', 'week_start'], inplace = True)

    # Get only user info
    user_info = post_df[['user_description', 'scree_name', 'regions', 'age', 'gender', 'is_org']]
    # drop duplicates
    user_info.drop_duplicates(subset = 'scree_name',inplace = True)
    assert user_info.shape[0] == post_df.scree_name.nunique()
    del post_df, pre_df
    # Now merge it with our pre-covid scrape
    pred = pred.merge(user_info, on = 'scree_name', how = 'outer', validate = 'm:1', indicator = True)
    assert pred._merge.value_counts()['left_only'] == 0

    # we drop unmerged values
    pred = pred.loc[pred._merge == 'both']
    pred.drop(columns = '_merge', inplace = True)

    # Sort all the values
    pred.sort_values(by= ['scree_name', 'week_start'], inplace = True)

    pred.to_pickle(f"{path_to_processed}df_processed.pkl.gz", compression = 'gzip')

# 2. Load Our BERT Model ---------------------------------------------------------------------

def padd_split(tokens, label, index, percentage_filter = 1):
  # split into chunks of 510 tokens, we also convert to list (default is tuple which is immutable)
  input_id_chunks = list(tokens['input_ids'][0].split(chunksize - 2))
  mask_chunks = list(tokens['attention_mask'][0].split(chunksize - 2))

  final_token = []
  # loop through each chunk
  for i in range(len(input_id_chunks)):
      single_token = {}
      # add CLS and SEP tokens to input IDs
      input_id_chunks[i] = torch.cat([
          torch.tensor([101]), input_id_chunks[i], torch.tensor([102])
      ])
      # add attention tokens to attention mask
      mask_chunks[i] = torch.cat([
          torch.tensor([1]), mask_chunks[i], torch.tensor([1])
      ])
      # get required padding length
      pad_len = chunksize - input_id_chunks[i].shape[0]
      # check if tensor length satisfies required chunk size
      if pad_len > 0:
          # if padding length is more than 0, we must add padding
          input_id_chunks[i] = torch.cat([
              input_id_chunks[i], torch.Tensor([0] * pad_len)
          ]).int()
          mask_chunks[i] = torch.cat([
              mask_chunks[i], torch.Tensor([0] * pad_len)
          ]).int()
      single_token['input_ids'] = input_id_chunks[i]
      single_token['attention_mask'] = mask_chunks[i]
      single_token['label'] = label
      single_token['index'] = index
      
      if percentage_filter <1 and (single_token['attention_mask'].sum()/len(single_token['attention_mask'])).item() >= percentage_filter:
        final_token.append(single_token)
      elif percentage_filter == 1:
        final_token.append(single_token)
  return final_token

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, percentage_filter = 1):
        tokenizer = AutoTokenizer.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")

        if merge_tweets:
            df = df.reset_index()
            self.texts = [padd_split(tokenizer.encode_plus(text, add_special_tokens = False, return_tensors="pt"), label, index, percentage_filter = percentage_filter) for text, label, index in zip(df['tweet_text'], df['final_polarization'], df.index)]
            self.texts = [token for text in self.texts for token in text]
            self.labels = [dictionary["label"] for dictionary in self.texts]
            self.index = [dictionary["index"] for dictionary in self.texts]
        else:
            self.texts = [tokenizer(text, padding='max_length', max_length = 128, truncation=True, return_tensors="pt") for text in df['tweet_text']]
            self.labels = [label for label in df['final_polarization']]
            self.index = [index for index in df.index]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])
  
    def get_batch_index(self, idx):
        # Fetch a batch of index
        return np.array(self.index[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        batch_index = self.get_batch_index(idx)

        return batch_texts, batch_y, batch_index



class BertClassifier(nn.Module):

    def __init__(self, dropout=0.2):

        super(BertClassifier, self).__init__()

        self.bert = AutoModel.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1) 
        self.tanh = nn.Tanh()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.tanh(linear_output)
        return final_layer

def mean_by_label(samples, labels):
    df = pd.DataFrame(samples, labels, columns = ['pred'])
    mean = df.groupby(df.index).pred.mean()
    return list(mean.values), list(mean.index)

def predict_from_model(model, data, batch_size = 32, bert_model = ''):

    dataloader = torch.utils.data.DataLoader(data, batch_size = batch_size)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    total_preds = []
    with torch.no_grad():
    
        for data_input, data_label, data_index in tqdm(dataloader):
          
            data_label = data_label.to(device)
            mask = data_input['attention_mask'].to(device)
            input_id = data_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            output = output.squeeze(-1)
            preds = output.detach().cpu().numpy()
            total_preds.extend(list(preds))

    mean_out, mean_index = mean_by_label(total_preds, data.index)
    total_preds = pd.DataFrame({'pred': total_preds, 'labels': data.labels}, index = data.index)
    for i, m_index in enumerate(mean_index):
      total_preds.loc[total_preds.index == mean_index[i], "pred"] = mean_out[i]
    total_preds = total_preds.groupby(total_preds.index).mean()

    return total_preds


model = BertClassifier()
torch.manual_seed(random_seed)

model.load_state_dict(torch.load(f'{path_to_models}best_torch_{learning_rate}_{batch_size}_{epoch_num}{percentage_filter}.pt', map_location = 'cpu'))

pred['prediction'] = 100
batch_dataset = Dataset(pred)
final_predictions = predict_from_model(model, batch_dataset, batch_size = 128, bert_model = '')
final_predictions.reset_index(inplace = True, drop = True)
pred['prediction'] = final_predictions.pred
final_predictions.to_pickle(f"{path_to_processed}final_df.pkl.gz", compression = 'gzip')