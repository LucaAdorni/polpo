###########################################################
# Code to produce examples for SHAP values
# Author: Luca Adorni
# Date: April 2023
###########################################################

# 0. Setup -------------------------------------------------

import numpy as np
import pandas as pd
import os
import sys
import pickle
import random
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import dill
import re
import shap
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, random_split
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Callable, Dict, Generator, List, Tuple, Union
from pathlib import PurePosixPath
import itertools
from torch import nn
from tqdm import tqdm

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
path_to_models = f"{path_to_data}models/"


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

# We slightly change the Dataset function to account for having individual tweets and not a dataframe
class Dataset2(torch.utils.data.Dataset):

    def __init__(self, text, percentage_filter = 1):
        tokenizer = AutoTokenizer.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")
        if type(text) == list or type(text) == np.ndarray:
          df = pd.DataFrame(text, columns = ['tweet_text'])
        else:
          df = pd.DataFrame([text], columns = ['tweet_text'])
        df['final_polarization'] = 100
        self.texts = [padd_split(tokenizer.encode_plus(text, add_special_tokens = False, return_tensors="pt"), label, index, percentage_filter = percentage_filter) for text, label, index in zip(df['tweet_text'], df['final_polarization'], df.index)]
        self.texts = [token for text in self.texts for token in text]
        self.labels = [dictionary["label"] for dictionary in self.texts]
        self.index = [dictionary["index"] for dictionary in self.texts]
 

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
    model.eval()
    with torch.no_grad():
    
        for data_input, data_label, data_index in dataloader:
          
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

# define a prediction function
def f(x):
  batch_dataset = Dataset2(x)
  p = predict_from_model(model, batch_dataset)
  return p.pred.tolist()

# Define a custom masker for SHAP.EXPLAINER
def custom_tokenizer(s, return_offsets_mapping=True):
    """ Custom tokenizers conform to a subset of the transformers API.
    """
    pos = 0
    offset_ranges = []
    input_ids = []
    for m in re.finditer(r"\W", s):
        start, end = m.span(0)
        offset_ranges.append((pos, start))
        input_ids.append(s[pos:start])
        pos = end
    if pos != len(s):
        offset_ranges.append((pos, len(s)))
        input_ids.append(s[pos:])
    out = {}
    out["input_ids"] = input_ids
    if return_offsets_mapping:
        out["offset_mapping"] = offset_ranges
    return out

# PARAMETERS ---------------
random_seed = 42
random.seed(random_seed)

bert_tag = '_256'
percentage_filter = 0.4
learning_rate = 3e-05
batch_size = 32
epoch_num = 3
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


# Import our model
model = BertClassifier()
torch.manual_seed(random_seed)
model.load_state_dict(torch.load(f'{path_to_models}best_torch_{learning_rate}_{batch_size}_{epoch_num}{percentage_filter}.pt', map_location = 'cpu'))

# Initialize SHAP Explainer

masker = shap.maskers.Text(custom_tokenizer)
explainer = shap.Explainer(f, masker)

# Load our final dataset
df = pd.read_pickle(f"{path_to_processed}final_df_clean.pkl.gz", compression = 'gzip')

# Get some relevant examples
t1 = df.loc[df.prediction > (df.prediction.max())*0.99].tweet_text.tolist()[29]
t2 = df.loc[df.prediction < (df.prediction.min())*0.99].tweet_text.tolist()[1]
t = [t1, t2]

# Get their explanation
shap_values = explainer(t)

# Save the Far-Right Example
plt.close()
shap.plots.waterfall(shap_values[0], show = False)
f = plt.gcf()
w = 10
f.set_size_inches(w, w*3/4)
plt.tight_layout()

plt.savefig(f'{path_to_figures}final/fig_a1_far_right_example.pdf', bbox_inches='tight',dpi=100)

# Save the Far-Left Example:
plt.close()
shap.plots.waterfall(shap_values[1], show = False)
f = plt.gcf()
w = 10
f.set_size_inches(w, w*3/4)
plt.tight_layout()

plt.savefig(f'{path_to_figures}final/fig_a1_far_left_example.pdf', bbox_inches='tight',dpi=100)