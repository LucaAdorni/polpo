###########################################################
# Code to train a BERT model for topic recognition
# Author: Luca Adorni
# Date: November 2022
###########################################################

# 0. Setup -------------------------------------------------
#!/usr/bin/env python
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD # initialize communications
    rank = comm.Get_rank()
    size = comm.Get_size()
except:
    print("Not running on a cluster")
    rank = 0


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

import transformers
from transformers import pipeline

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

  
print(path_to_repo)

path_to_data = f"{path_to_repo}data/"
path_to_figures = f"{path_to_repo}figures/"
path_to_raw = f"{path_to_data}raw/"
path_to_links = f"{path_to_data}links/"
path_to_topic = f"{path_to_data}topic/"
path_to_results = f"{path_to_data}results/"
path_to_models = f"{path_to_data}models/"
path_to_models_top = f"{path_to_models}topic_class/"
path_to_alberto = "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0"

os.makedirs(path_to_figures, exist_ok = True)
os.makedirs(path_to_topic, exist_ok = True)
os.makedirs(path_to_results, exist_ok = True)
os.makedirs(path_to_models, exist_ok = True)
os.makedirs(path_to_models_top, exist_ok = True)

# 2. Read Files --------------------------------------------------------------------------------------------------

print("Process - Rank: %d -----------------------------------------------------"%rank)


train = pd.read_pickle(f"{path_to_topic}train.pkl.gz", compression = 'gzip').head()
val = pd.read_pickle(f"{path_to_topic}val.pkl.gz", compression = 'gzip').head()

# Transform our topics into labels
train.topic.replace({'economics': 1, 'politics': 2, 'art and entertainment': 0, 'health': 0, 'vaccine': 0, 'none': 0}, inplace = True)
val.topic.replace({'economics': 1, 'politics': 2, 'art and entertainment': 0, 'health': 0, 'vaccine': 0, 'none': 0}, inplace = True)


# 3. Define our BERT Module --------------------------------------------------------------------------------------------------

# reference https://github.com/kswamy15/pytorch-lightning-imdb-bert/blob/master/Bert_NLP_Pytorch_IMDB_v3.ipynb

# custom dataset uses Bert Tokenizer to create the Pytorch Dataset
class TopicDataset(Dataset):
    def __init__(self, tweets, targets, tokenizer, max_len):
        self.tweets = tweets
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
         
    def __len__(self):
        return (len(self.tweets))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        tweets = str(self.tweets[idx])
        target = self.targets[idx]
        
        encoding = self.tokenizer.encode_plus(
          tweets,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=True,
          truncation=True,
          padding='max_length',
          return_attention_mask=True,
          return_tensors='pt',
        )    
        return {
            'label': torch.tensor(target, dtype=torch.long),
            'input_ids': (encoding['input_ids']).flatten(),
            'attention_mask': (encoding['attention_mask']).flatten(),
            'token_type_ids': (encoding['token_type_ids']).flatten()
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
  
  main_dir = f"{path_to_models_top}lr{learning_rate}_batch{batch_size}/"
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
                 num_labels: int = 3,
                 **kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        self._add_model_specific_hparams()
        self._add_default_hparams()
        self._set_seed(self.hparams.random_seed)
        # Adjust the output path
        self.hparams.output_path = f"{path_to_models_top}lr{self.hparams.learning_rate}_batch{self.hparams.batch_size}/"
        os.makedirs(self.hparams.output_path, exist_ok = True)
        
        self.num_labels = num_labels
   
        config = AutoConfig.from_pretrained(path_to_alberto)
        self.bert = AutoModel.from_pretrained(path_to_alberto)
        
        self.pre_classifier = torch.nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout = torch.nn.Dropout(0.2)

        # relu activation function
        self.relu =  torch.nn.ReLU()

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
            "loader_workers": multiprocessing.cpu_count(),
            "output_path": f"{path_to_models_top}",
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
    df_cm = pd.DataFrame(cm, columns=["0","1", "2"], index = ["0","1", "2"])
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


# Need to get the correct emotion
def get_max_emotion(x):
    store = []
    eval = []
    for pred in x:
        store.append(pred['label'])
        eval.append(pred['score'])
    pred_f = store[np.argmax(eval)]
    return pred_f

# Load the FEEL-IT Model
classifier = pipeline("text-classification",model='MilaNLProc/feel-it-italian-emotion',top_k=2)

# Define our main tokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/adorni/polpo/alberto/tokenizer/models--m-polignano-uniba--bert_uncased_L-12_H-768_A-12_italian_alb3rt0/snapshots/4454cfbc82952da79729e33e81c37a72dc095b4b")

use_cuda = torch.cuda.is_available()
print(f"Is CUDA Available? {use_cuda}")

# 4. Initialize our models --------------------------------------------------------------------------------------------------

# Remove index
train.reset_index(inplace = True, drop = True)
val.reset_index(inplace = True, drop = True)

# Define our datasets
train_dataset  = TopicDataset(tweets = train['tweet_text'], targets = train['topic'], tokenizer = tokenizer, max_len = 128)
val_dataset  = TopicDataset(tweets = val['tweet_text'], targets = val['topic'], tokenizer = tokenizer, max_len = 128)


# Initialize a list to store all our results
try:
    with open(f'{path_to_results}topic_performance.pkl', "rb") as fp:   # Unpickling
        pred_performance = pickle.load(fp)
        print("Loaded pre-existing results")
except:
    pred_performance = {}

# 5. Predict over the whole Pre-Covid dataset --------------------------------------------------------------------------------------------------

# First we check what is the best model
models = []
f1 = []
f1_gap = []
acc = []
prec = []
recall = []

for k,v in pred_performance.items():
    print(k)
    print(v)
    models.append(k)
    f1.append(v['f1'])
    f1_gap.append(v['f1_gap'])
    acc.append(v['acc'])
    prec.append(v['prec'])
    recall.append(v['recall'])

pred_performance
# Take the maximum of the F1-score (macro-avg) as the best model
chosen_model = np.argmax(f1)

# Get the correct key
best_model = models[chosen_model]
assert np.max(f1) == pred_performance[best_model]['f1']

lr = re.findall("lr_(.*?)_batch_[0-9]*", best_model)[0]
batch = re.sub("lr_[0-9]*e-[0-9]*_batch_","", best_model)

# Get the correct path out of the best model
path = get_best_checkpoint_path(batch_size = batch, learning_rate = lr)[0]

# We need to define the test_dataset to initialize the model
test_dataset = TopicDataset(tweets = train['tweet_text'], targets = train['topic'], tokenizer = tokenizer, max_len = 128)

# Load from the best path
model = BertModule.load_from_checkpoint(f"{path}")
model.hparams['loader_workers'] = round(multiprocessing.cpu_count()*0.8)
model.hparams['batch_size'] = 128

# Again iteratively load only portion of the dataset
# PICK MAX RANGE AS TOT SIZE OF BATCHES / N.of Machines
max_range = 2
n_machines = 125

for iter in range(0,max_range):
    if iter != 0:
        rank = rank + n_machines
    # Load the scraped pre-covid tweets
    try:
        scraped_df = pd.read_pickle(f'{path_to_raw}batches/batch_{rank}.pkl.gz', compression='gzip')

        if "final_pred" in scraped_df.columns:
            continue
        else:
            # Define an empty target column
            scraped_df['topic'] = 100

            # Reset the index
            scraped_df.reset_index(inplace = True, drop = True)

            # Define our datasets
            model.data.test_dataset = TopicDataset(tweets = scraped_df['tweet_text'], targets = scraped_df['topic'], tokenizer = tokenizer, max_len = 128)

            # Define the trainer
            trainer_pred = Trainer()
            # Then start predicting
            pred = trainer_pred.predict(model, model.data.test_dataloader())

            # First we flatten our list of tensors into a list of predictions
            pred = [y_hat for tensor in pred for y_hat in tensor.tolist()]

            # Then we transform it into a dataframe
            pred = pd.DataFrame(pred, columns = ['topic_0', 'topic_1', 'topic_2'])
            # Now we calculate the correct topic
            pred['final_pred'] = pred.apply(lambda x: np.argmax(x), axis = 1)

            # Merge the two datasets
            scraped_df.reset_index(inplace = True, drop = True)
            pred.reset_index(inplace = True, drop = True)
            scraped_df = pd.concat([scraped_df, pred], axis = 1)

            # Use FEEL-IT to predict emotions
            em_pred = classifier(batch.tweet_text.tolist())
            scraped_df['emotion'] = em_pred
            scraped_df['emotion'] = scraped_df.emotion.apply(lambda x: get_max_emotion(x))

            # Then save it
            scraped_df.reset_index(inplace = True, drop = True)
            scraped_df.to_pickle(f"{path_to_raw}batches/batch_{rank}.pkl.gz", compression = 'gzip')

            del pred, scraped_df, trainer_pred
    except:
        continue