###########################################################
# Code to train a BERT model for polarization prediction
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
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, random_split
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Callable, Dict, Generator, List, Tuple, Union
from pathlib import PurePosixPath
import itertools
from torch import nn

try:
    # Setup Repository
    with open("repo_info.txt", "r") as repo_info:
        path_to_repo = repo_info.readline()
except:
    path_to_repo = f"{os.getcwd()}/polpo/"
    sys.path.append(f"{os.getcwd()}/.local/bin") # import the temporary path where the server installed the module

# PARAMETERS------------------
classification = False

# MAIN PARAMETERS
max_len = 256
percentage_filter = 0.4
# Loop over a variety of Parameters
learning_rates = [3e-5]
batch_size = [32]
EPOCHS = 6

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


parameters = list(itertools.product(learning_rates, batch_size))
  
print(path_to_repo)

path_to_data = f"{path_to_repo}data/"
path_to_figures = f"{path_to_repo}figures/"
path_to_raw = f"{path_to_data}raw/"
path_to_links = f"{path_to_data}links/"
path_to_topic = f"{path_to_data}topic/"
path_to_results = f"{path_to_data}results/"
path_to_models = f"{path_to_data}models/"
if classification:
    path_to_models_pol = f"{path_to_models}pol_class{merged_tag}{percentage_filter}/"
else:
    path_to_models_pol = f"{path_to_models}pol_reg{merged_tag}{percentage_filter}/"
path_to_processed = f"{path_to_data}processed/"
path_to_alberto = "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0"

os.makedirs(path_to_figures, exist_ok = True)
os.makedirs(path_to_topic, exist_ok = True)
os.makedirs(path_to_results, exist_ok = True)
os.makedirs(path_to_models, exist_ok = True)
os.makedirs(path_to_models_pol, exist_ok = True)

# 2. Read Files --------------------------------------------------------------------------------------------------

train = pd.read_pickle(f"{path_to_processed}df_train{merged_tag}.pkl.gz", compression = 'gzip')
val = pd.read_pickle(f"{path_to_processed}df_val{merged_tag}.pkl.gz", compression = 'gzip')
test = pd.read_pickle(f"{path_to_processed}df_test{merged_tag}.pkl.gz", compression = 'gzip')

print(f"\nNumber of classes: {train.polarization_bin.nunique()}")

# Load our AlBERTo model
bert_model = AutoModel.from_pretrained("/home/adorni/polpo/data/alberto_custom")

chunksize = 256

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
        tokenizer = AutoTokenizer.from_pretrained("/home/adorni/polpo/data/alberto_custom")

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

        self.bert = AutoModel.from_pretrained("/home/adorni/polpo/data/alberto_custom")
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
    mean = df.groupby(df.index).mean()
    return list(mean.loc[:,'pred']), list(mean.index)

def predict_from_model(model, data, batch_size = 32, bert_model = ''):

    dataloader = torch.utils.data.DataLoader(data, batch_size = batch_size)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    model.eval()
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

    return total_preds

def evaluate(model, test, batch_size = 32, bert_model = ''):


    total_preds = predict_from_model(model, test, batch_size = batch_size, bert_model = bert_model)

    if merge_tweets:
      mean_out, mean_index = mean_by_label(total_preds, test.index)
      total_preds = pd.DataFrame({'pred': total_preds, 'labels': test.labels}, index = test.index)
      for i, m_index in enumerate(mean_index):
        total_preds.loc[total_preds.index == mean_index[i], "pred"] = mean_out[i]
      total_preds = total_preds.groupby(total_preds.index).mean()


    mae = mean_absolute_error(total_preds.labels, total_preds.pred)
    mse = mean_squared_error(total_preds.labels, total_preds.pred)
    r_squared = r2_score(total_preds.labels, total_preds.pred)
    
    print(f'MAE: {mae: .3f}')
    print(f'MSE: {mse: .3f}')
    print(f'R2: {r_squared: .3f}')

    test_final = {'MAE': mae, 'MSE': mse, 'R2': r_squared}

    return test_final

def train_model(model, train, val, test, learning_rate, epochs, batch_size = 32, bert_model = '', old_mae = 0.3):

    results_train = {}
    results_val = {}
    results_test = {}

    use_cuda = torch.cuda.is_available()
    print(f"Is CUDA Available? {use_cuda}")
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr= learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    for epoch_num in range(epochs):
            print(f'Epoch: {epoch_num + 1}')
            
            total_loss_train = 0

            for train_input, train_label, train_index in tqdm(train_dataloader):
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)   

                train_label = train_label.to(device)             

                train_label = train_label.float()                

                output = model(input_id, mask)   
               
                output = output.squeeze(-1)

                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()
                

                model.zero_grad()

                batch_loss.backward()

                optimizer.step()
          

            #train_final = evaluate(model, train, batch_size = batch_size, bert_model = bert_model)
            val_final = evaluate(model, val, batch_size = batch_size, bert_model = bert_model)
            test_final = evaluate(model, test, batch_size = batch_size, bert_model = bert_model)
            

            print(f'Epochs: {epoch_num + 1} Train Loss: {total_loss_train: .3f}\n \
            Val: \| MAE: {val_final["MAE"]} \| MSE: {val_final["MSE"]} \| R2: {val_final["R2"]} \n \
            Test MAE: {test_final["MAE"]} \| Test MSE: {test_final["MSE"]} \| Test R2: {test_final["R2"]}')
            new_mae = val_final['MAE']
            check_mae = new_mae < old_mae

            results_val[f'Epoch{epoch_num+1}'] = val_final
            results_test[f'Epoch{epoch_num+1}'] = test_final

            if check_mae:
                print(f'iteration got us a good mae :)')
                torch.save(model.state_dict(), f'{path_to_models}best_torch_{learning_rate}_{batch_size}_{epoch_num}{percentage_filter}.pt')
                old_mae = new_mae

    return results_val, results_test, old_mae

LR = 3e-5
batch_sz = 32
old_mae = 0.3
random_seed = 42

train = Dataset(train, percentage_filter = percentage_filter)
val = Dataset(val, percentage_filter = percentage_filter)
test = Dataset(test, percentage_filter = percentage_filter)


model = BertClassifier()
path_to_embed = path_to_alberto
torch.manual_seed(random_seed) # set our seed
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
val_results, test_results, old_mae = train_model(model, train, val, test, LR, EPOCHS, batch_size = batch_sz, bert_model = bert_model, old_mae = old_mae)