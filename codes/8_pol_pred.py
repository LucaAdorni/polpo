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
import multiprocessing
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
#from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer, seed_everything
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Callable, Dict, Generator, List, Tuple, Union
from pathlib import PurePosixPath
import itertools

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

try:
    # Setup Repository
    with open("repo_info.txt", "r") as repo_info:
        path_to_repo = repo_info.readline()
except:
    path_to_repo = f"{os.getcwd()}/polpo/"
    sys.path.append(f"{os.getcwd()}/.local/bin") # import the temporary path where the server installed the module


# PARAMETERS------------------
classification = True
  
print(path_to_repo)

path_to_data = f"{path_to_repo}data/"
path_to_figures = f"{path_to_repo}figures/"
path_to_raw = f"{path_to_data}raw/"
path_to_links = f"{path_to_data}links/"
path_to_topic = f"{path_to_data}topic/"
path_to_results = f"{path_to_data}results/"
path_to_models = f"{path_to_data}models/"
if classification:
    path_to_models_pol = f"{path_to_models}pol_class/"
else:
    path_to_models_pol = f"{path_to_models}pol_reg/"
path_to_processed = f"{path_to_data}processed/"
path_to_alberto = "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0"

os.makedirs(path_to_figures, exist_ok = True)
os.makedirs(path_to_topic, exist_ok = True)
os.makedirs(path_to_results, exist_ok = True)
os.makedirs(path_to_models, exist_ok = True)
os.makedirs(path_to_models_pol, exist_ok = True)

# 2. Read Files --------------------------------------------------------------------------------------------------

train = pd.read_pickle(f"{path_to_processed}df_train.pkl.gz", compression = 'gzip')
val = pd.read_pickle(f"{path_to_processed}df_val.pkl.gz", compression = 'gzip')
test = pd.read_pickle(f"{path_to_processed}df_test.pkl.gz", compression = 'gzip')

# Transform our topics into labels
replace_dict = {"far_left": 0, "center_left": 1, "center":2, "center_right":3, "far_right":4}
train.polarization_bin.replace(replace_dict, inplace = True)
val.polarization_bin.replace(replace_dict, inplace = True)
test.polarization_bin.replace(replace_dict, inplace = True)
print(f"\nNumber of classes: {train.polarization_bin.nunique()}")

# 3. Define our BERT Module --------------------------------------------------------------------------------------------------

# reference https://github.com/kswamy15/pytorch-lightning-imdb-bert/blob/master/Bert_NLP_Pytorch_IMDB_v3.ipynb

# Since we merged tweets, need to split them into tensors of maximum 512 length
def padd_split(tokens, label, index, percentage_filter = 1, chunksize = 256):
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


class TopicDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, percentage_filter = 1):      
        df = df.reset_index()
        self.tokenizer = tokenizer
        self.max_len = max_len
        if classification:
            targ_name = "polarization_bin"
        else:
            targ_name = "final_polarization"
        self.texts = [padd_split(self.tokenizer.encode_plus(text, add_special_tokens = False, return_token_type_ids=True, return_tensors="pt"), label, index, percentage_filter = percentage_filter, chunksize = max_len) for text, label, index in zip(df['tweet_text'], df[targ_name], df.index)]
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

        return {
            'label': torch.tensor(batch_y, dtype=torch.long),
            'input_ids': (batch_texts['input_ids']).flatten(),
            'attention_mask': (batch_texts['attention_mask']).flatten(),
            #'token_type_ids': (batch_texts['token_type_ids']).flatten(),
            'index': batch_index
        }


# Define a function to get the best checkpoint
def get_best_checkpoint_path(
      batch_size: float,
      learning_rate: float,
      model_class: pl.LightningModule = None,
      metric: str = "avg_val_loss",
      asc: bool = True,
      file_format: str = ".ckpt",
  ) -> str:    
  
  main_dir = f"{path_to_models_pol}lr{learning_rate}_batch{batch_size}/"
  checkpoints = [
      PurePosixPath(f"{main_dir}{el}")
      for el in os.listdir(main_dir)
      if os.path.isfile(path = PurePosixPath(f"{main_dir}{el}")) and PurePosixPath(f"{main_dir}{el}").suffix == ".ckpt" and f"batch={batch_size}-lr={str(learning_rate)}" in PurePosixPath(f"{main_dir}{el}").stem
  ]
  path_score = {
      str(path): float(path.stem.split("=")[-1])
      for path in checkpoints
      if metric in path.stem
  }
  epoch_n = {
      str(path): float(re.search("epoch=(.+?)-", path.stem).groups()[0])
      for path in checkpoints
      if metric in path.stem
  }
  if asc:
      best_checkpoint_path = min(path_score, key=path_score.get)
  else:
      best_checkpoint_path = max(path_score, key=path_score.get)
  return best_checkpoint_path, epoch_n[best_checkpoint_path]

# Now we define our Pytorch Lightning module
## The main Pytorch Lightning module
class BertModule(pl.LightningModule):
    class DataModule(pl.LightningDataModule):
            def __init__(self, model_instance: pl.LightningModule) -> None:
                super().__init__()
                self.hparams.update(model_instance.hparams)
                self.model = model_instance
                self.generator = torch.Generator()
                self.generator.manual_seed(self.model.hparams.random_seed)

                # initialize datasets
                self.train_dataset = train_dataset
                self.dev_dataset = val_dataset
                self.test_dataset = test_dataset

            def train_dataloader(self) -> DataLoader:
                return DataLoader(
                    dataset=self.train_dataset,
                    shuffle=self.hparams.shuffle_train_dataset,
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.loader_workers,
                    worker_init_fn=self._seed_worker,
                    generator=self.generator,
                )

            def val_dataloader(self) -> DataLoader:
                return DataLoader(
                    dataset=self.dev_dataset,
                    shuffle=False,
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.loader_workers,
                    worker_init_fn=self._seed_worker,
                    generator=self.generator,
                )

            def test_dataloader(self) -> DataLoader:
                return DataLoader(
                    dataset=self.test_dataset,
                    shuffle=False,
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.loader_workers,
                    worker_init_fn=self._seed_worker,
                    generator=self.generator,
                )

            def _seed_worker(self, worker_id):
                worker_seed = torch.initial_seed() % 2**32
                np.random.seed(worker_seed)
                random.seed(worker_seed)


    def __init__(self,
                 num_labels: int = 5,
                 **kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        self._add_model_specific_hparams()
        self._add_default_hparams()
        self._set_seed(self.hparams.random_seed)
        # Adjust the output path
        self.hparams.output_path = f"{path_to_models_pol}lr{self.hparams.learning_rate}_batch{self.hparams.batch_size}/"
        os.makedirs(self.hparams.output_path, exist_ok = True)
        if classification:
            self.num_labels = num_labels
        else:
            self.num_labels = 1
   
        config = AutoConfig.from_pretrained(path_to_alberto)
        self.bert = AutoModel.from_pretrained(path_to_alberto)
        
        self.pre_classifier = torch.nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout = torch.nn.Dropout(0.2)

        # relu activation function
        self.relu =  torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()
        self.tanh = torch.nn.Tanh()

        # Build DataModule
        self.data = self.DataModule(self)

        self.trainer_params = self._get_trainer_params()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        input_ids = batch['input_ids']
        label = batch['label']
        attention_mask = batch['attention_mask']
        #token_type_ids = batch['token_type_ids']
        y_hat = self(input_ids, attention_mask, label)
        return y_hat

    def forward(self, input_ids, attention_mask, labels):
      
        outputs = self.bert(input_ids=input_ids, \
                         attention_mask=attention_mask)

        hidden_state = outputs[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = self.relu(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)
        if classification:
            logits = self.softmax(logits)
        else:
            logits = self.tanh(logits)

        return logits

    def get_outputs(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, \
                         attention_mask=attention_mask)
        hidden_state = outputs[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        return pooled_output
        

    def training_step(self, batch, batch_nb) -> torch.Tensor:
        # batch
        input_ids = batch['input_ids']
        label = batch['label']
        attention_mask = batch['attention_mask']
        # fwd
        y_hat = self(input_ids, attention_mask, label)
        
        # loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(y_hat.view(-1, self.num_labels), label.view(-1))

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.hparams.batch_size,
        )

        return loss

    def training_epoch_end(self, training_step_outputs) -> None:
        self.avg_train_loss = torch.Tensor(
            self._stack_outputs(training_step_outputs)
        ).mean()  # stored in order to be accessed by Callbacks
        self.log("avg_train_loss", self.avg_train_loss, logger=True)


    def validation_step(self, batch, batch_nb) -> torch.Tensor:
        # batch
        input_ids = batch['input_ids']
        label = batch['label']
        attention_mask = batch['attention_mask']
        # fwd
        y_hat = self(input_ids, attention_mask, label)
        
        # loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(y_hat.view(-1, self.num_labels), label.view(-1))

        # acc
        a, y_hat = torch.max(y_hat, dim=1)
        val_acc = accuracy_score(y_hat.cpu(), label.cpu())
        val_acc = torch.tensor(val_acc)

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.hparams.batch_size,
        )
        return loss

    def validation_epoch_end(self, val_step_outputs) -> None:
        self.avg_val_loss = torch.stack(
            tuple(val_step_outputs)
        ).mean()  # stored in order to be accessed by Callbacks
        self.log("avg_val_loss", self.avg_val_loss, logger=True)
    
    def on_batch_end(self):
        #for group in self.optim.param_groups:
        #    print('learning rate', group['lr'])
        # This is needed to use the One Cycle learning rate that needs the learning rate to change after every batch
        # Without this, the learning rate will only change after every epoch
        if self.sched is not None:
            self.sched.step()

    def test_step(self, batch, batch_nb):
        input_ids = batch['input_ids']
        label = batch['label']
        attention_mask = batch['attention_mask']
        #token_type_ids = batch['token_type_ids']
        y_hat = self(input_ids, attention_mask, label)
        
        # loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(y_hat.view(-1, self.num_labels), label.view(-1))
        
        a, y_hat = torch.max(y_hat, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), label.cpu())

        self.log(
            "test_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.hparams.batch_size,
        )
        return loss
        

    def test_epoch_end(self, outputs) -> None:
        self.avg_test_loss = torch.stack(
            tuple(outputs)
        ).mean()  # stored in order to be accessed by Callbacks
        self.log("avg_test_loss", self.avg_test_loss, logger=True)

    
    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        # REQUIRED

        optimizer = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.hparams.learning_rate, eps=1e-08, weight_decay=self.hparams.optimizer_weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.hparams.lr_scheduler_factor,
            patience=self.hparams.lr_scheduler_patience,
            min_lr=self.hparams.lr_scheduler_min_lr,
        )

        optim_config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "avg_val_loss",
            },
        }

        return optim_config

    def fit(self) -> None:
        self._set_seed(self.hparams.random_seed)
        self.trainer = Trainer(**self.trainer_params)
        self.trainer.fit(self, datamodule=self.data)

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seed_everything(seed)  # Pytorch Lightning function

    def _stack_outputs(self, outputs) -> torch.Tensor:
        if isinstance(outputs, list):
            return [self._stack_outputs(output) for output in outputs]
        elif isinstance(outputs, dict):
            return outputs["loss"]
    
    def _get_trainer_params(self) -> Dict:

        backup_callback = ModelCheckpoint(
            dirpath=self.hparams.output_path,
            filename= f"batch={self.hparams.batch_size}-lr={str(self.hparams.learning_rate)}" + "backup-{epoch}-{avg_val_loss:.2f}",
            every_n_epochs=self.hparams.backup_n_epochs,
            save_on_train_epoch_end=True,
            verbose=self.hparams.verbose,
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.hparams.output_path,
            filename= f"batch={self.hparams.batch_size}-lr={str(self.hparams.learning_rate)}" + "{epoch}-{avg_val_loss:.2f}",
            monitor=self.hparams.checkpoint_monitor,
            mode=self.hparams.checkpoint_monitor_mode,
            verbose=self.hparams.verbose,
        )

        early_stop_callback = EarlyStopping(
            monitor=self.hparams.early_stop_monitor,
            min_delta=self.hparams.early_stop_min_delta,
            patience=self.hparams.early_stop_patience,
            verbose=self.hparams.verbose,
        )

        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        callbacks = [
            # backup_callback,
            checkpoint_callback,
            early_stop_callback,
            lr_monitor
        ]

        trainer_params = {
            "callbacks": callbacks,
            "default_root_dir": self.hparams.output_path,
            "accumulate_grad_batches": self.hparams.accumulate_grad_batches,
            "accelerator": self.hparams.accelerator,
            "devices": self.hparams.devices,
            "max_epochs": self.hparams.max_epochs,
            "deterministic": self.hparams.deterministic
        }

        return trainer_params

    def get_best_path(self, **kwargs) -> pl.LightningModule:
        best_checkpoint_path, _ = get_best_checkpoint_path(self.hparams.batch_size, self.hparams.learning_rate, model_class=self, **kwargs)
        return best_checkpoint_path

    def load_from_best_checkpoint(self, **kwargs) -> pl.LightningModule:
        best_checkpoint_path, _ = get_best_checkpoint_path(self.hparams.batch_size, self.hparams.learning_rate, model_class=self, **kwargs)
        print(best_checkpoint_path)
        return self.load_from_checkpoint(checkpoint_path=best_checkpoint_path)

    def _add_default_hparams(self) -> None:
        default_params = {
            "random_seed": 42,
            "deterministic": True,
            "shuffle_train_dataset": True,
            "batch_size": 32,
            "loader_workers": 0,
            "output_path": f"{path_to_models_pol}",
            # Trainer params
            "verbose": True,
            "accumulate_grad_batches": 1,
            "accelerator": 'gpu',
            "devices": 1,
            "max_epochs": 6,
            # Callback params
            "checkpoint_monitor": "avg_val_loss",
            "checkpoint_monitor_mode": "min",
            "early_stop_monitor": "avg_val_loss",
            "early_stop_min_delta": 0,
            "early_stop_patience": 10,
            "backup_n_epochs": 5,
            # Optimizer params
            'learning_rate': 5e-05,
            "optimizer_name": "adamw",
            "optimizer_weight_decay": 1e-3,
            "lr_scheduler_factor": 0.2,
            "lr_scheduler_patience": 2,
            "lr_scheduler_min_lr": 1e-7,
        }
        self.hparams.update({**default_params, **self.hparams})

    def _add_model_specific_hparams(self) -> None:
        pass


# Function to get the mean prediction
def mean_by_label(samples, labels):
    if classification:
        df = pd.DataFrame(samples, labels, columns = [0, 1, 2, 3, 4])
    else:
        df = pd.DataFrame(samples, labels, columns = [0])
    mean = df.groupby(df.index).mean()
    mean['y_hat'] = mean.idxmax(axis = 1)
    return mean

# define a function that saves figures
def save_fig(fig_id, tight_layout=True):
    # The path of the figures folder ./Figures/fig_id.png (fig_id is a variable that you specify 
    # when you call the function)
    path = os.path.join(path_to_repo,"figures", fig_id + ".pdf") 
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='pdf', dpi=300)
    
# Code to plot the confusion matrix and the main scores of our model

def model_scores_multiclass(y, y_hat, name = ''):
    """
    Plot the scores for a multiclass model (e.g. OVO/OVR)
    model: our multiclass model
    X: our predictors
    y: true values of y
    name: specify the name if we want to save the figure
    """
    if classification:
        f1score = f1_score(y,y_hat, average = 'macro')
        f1score_all = f1_score(y,y_hat, average = None)
        f1score_gap = max(f1score_all)- min(f1score_all)
        prec = precision_score(y, y_hat, average = 'macro')
        recall = recall_score(y, y_hat, average = 'macro')
        accuracy = accuracy_score(y, y_hat)
        print("F1-Score Macro = {}".format(round(f1score,5)))
        print("F1-Score Gap = {}".format(round(f1score_gap,5)))
        print("Accuracy = {}".format(round(accuracy,5)))
        print("")
        print(classification_report(y,y_hat,digits=5))
        cm = confusion_matrix(y, y_hat)
        party_list = ["far_left","center_left", "center", "center_right", "far_right"]
        df_cm = pd.DataFrame(cm, columns=party_list, index = party_list)
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        plt.figure(figsize = (10,8))
        sns.set(font_scale=1.4)#for label size
        sns.heatmap(df_cm, cmap="Blues", annot=True, fmt='g',annot_kws={"size": 16})# font size
        if name != '': 
            plt.title(name)
            save_fig(name)
        plt.show() 
        return {'f1':f1score, 'f1_gap': f1score_gap, 'acc': accuracy, 'prec': prec, 'recall': recall}
    else:
        mae = mean_absolute_error(y, y_hat)
        mse = mean_squared_error(y, y_hat)
        r_squared = r2_score(y, y_hat)
        return {'MAE': mae, 'MSE': mse, 'R2': r_squared}


# Define our main tokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/adorni/polpo/alberto/tokenizer/models--m-polignano-uniba--bert_uncased_L-12_H-768_A-12_italian_alb3rt0/snapshots/4454cfbc82952da79729e33e81c37a72dc095b4b")

use_cuda = torch.cuda.is_available()
print(f"Is CUDA Available? {use_cuda}")

# 4. Train our models --------------------------------------------------------------------------------------------------

# MAIN PARAMETERS
max_len = 256
percentage_filter = 1

# Remove index
train.reset_index(inplace = True, drop = True)
val.reset_index(inplace = True, drop = True)
test.reset_index(inplace = True, drop = True)

# Define our datasets
train_dataset  = TopicDataset(df = train, tokenizer = tokenizer, max_len = max_len, percentage_filter = percentage_filter)
val_dataset  = TopicDataset(df = val, tokenizer = tokenizer, max_len = max_len, percentage_filter = percentage_filter)
test_dataset  = TopicDataset(df = test, tokenizer = tokenizer, max_len = max_len, percentage_filter = percentage_filter)

# Loop over a variety of Parameters
learning_rates = [3e-5, 5e-5]
batch_size = [32, 16]

parameters = list(itertools.product(learning_rates, batch_size))

if classification:
    class_tag = ""  
    class_name = ""
else:
    class_tag = "_reg"
    class_name = "-Reg"

# Initialize a list to store all our results
try:
    with open(f'{path_to_results}polarization_performance{class_tag}.pkl', "rb") as fp:   # Unpickling
        pred_performance = pickle.load(fp)
        print("Loaded pre-existing results")
except:
    pred_performance = {}

# Loop and train a variety of BERT Models:
for lr, batch in parameters:
    if f'lr_{str(lr)}_batch_{batch}' in pred_performance:
        continue
    else:
        print("-"*100)
        print(f"\nTraining model with: learning rate: {lr}, batch size: {batch}\n")
        print("-"*100)
        # Fit the model
        try:
            # Initialize the model
            path = get_best_checkpoint_path(batch_size = batch, learning_rate = lr)[0]
            print("Model already trained")
        except:  
            model = BertModule(learning_rate = lr, already_trained = False, batch_size = batch)
            print("Fitting new model")
            model.fit()
            # Now within the same parameter instance, load the model and get the performance over the test set
            del model
            path = get_best_checkpoint_path(batch_size = batch, learning_rate = lr)[0]

        # Load from the best path
        model = BertModule.load_from_checkpoint(f"{path}")
        trainer_pred = Trainer()
        pred = trainer_pred.predict(model, model.data.test_dataloader())
        pred = [y_hat for tensor in pred for y_hat in tensor.tolist()]
        mean_pred = mean_by_label(pred, test_dataset.index)
        # Now plot the performances of our model
        pred_performance[f'lr_{str(lr)}_batch_{batch}'] = model_scores_multiclass(test.polarization_bin, mean_pred.y_hat, name = f'BERT-Polarization{class_name}-lr={str(lr)}-batch={batch}')
        # Delete the model to save memory
        del model
        torch.cuda.empty_cache() # PyTorch thing
    # Now save all the performances of our models
    with open(f'{path_to_results}polarization_performance{class_tag}.pkl', 'wb') as f:
        pickle.dump(pred_performance, f)