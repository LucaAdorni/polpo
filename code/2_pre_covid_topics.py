###########################################################
# Code to train a BERT model for topic recognition
# Author: Luca Adorni
# Date: October 2022
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
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer, seed_everything
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Callable, Dict, Generator, List, Tuple, Union

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

path_to_data = f"{path_to_repo}data/"
path_to_raw = f"{path_to_data}raw/"
path_to_links = f"{path_to_data}links/"
path_to_topic = os.path.join(path_to_data, "topic/","")

os.makedirs(path_to_topic, exist_ok = True)

# 2. Read Files --------------------------------------------------------------------------------------------------

print("Process - Rank: %d -----------------------------------------------------"%rank)

try:
    train = pd.read_pickle(f"{path_to_topic}train.pkl.gz", compression = 'gzip')
    val = pd.read_pickle(f"{path_to_topic}val.pkl.gz", compression = 'gzip')
    test = pd.read_pickle(f"{path_to_topic}test.pkl.gz", compression = 'gzip')
except:

    # Import our original dataset
    post_df = pd.read_csv(f'{path_to_raw}tweets_without_duplicates_regions_sentiment_demographics_topic_emotion.csv')
    # Drop columns we do not need
    post_df = post_df[['tweet_text', 'topic']]

    # PARAMETERS
    unused_size = 0.98
    test_size = 0.1
    val_size = 0.1
    train_size = 1 - test_size - val_size
    tag_size = '_02'
    random_seed = 42
    random.seed(random_seed)

    # Perform split into train and test
    train, _ = train_test_split(post_df, test_size = unused_size, random_state = random_seed, stratify = post_df.topic)
    train, test = train_test_split(train, test_size = test_size, random_state = random_seed, stratify = train.topic)
    train, val = train_test_split(train, test_size = val_size, random_state = random_seed, stratify = train.topic)

    # Print dataset shapes
    print(f'Training Set: {train.shape}')
    print(f'Validation Set: {val.shape}')
    print(f'Test Set: {test.shape}')

    # save the file
    train.to_pickle(f"{path_to_topic}train.pkl.gz", compression = 'gzip')
    val.to_pickle(f"{path_to_topic}val.pkl.gz", compression = 'gzip')
    test.to_pickle(f"{path_to_topic}test.pkl.gz", compression = 'gzip')
    print("Dataframes successfully saved")

# Transform our topics into labels
train.topic.replace({'economics': 1, 'politics': 2, 'art and entertainment': 0, 'health': 0, 'vaccine': 0, 'none': 0}, inplace = True)
val.topic.replace({'economics': 1, 'politics': 2, 'art and entertainment': 0, 'health': 0, 'vaccine': 0, 'none': 0}, inplace = True)
test.topic.replace({'economics': 1, 'politics': 2, 'art and entertainment': 0, 'health': 0, 'vaccine': 0, 'none': 0}, inplace = True)
print(train.topic.nunique())

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

# Define our main tokenizer
tokenizer = AutoTokenizer.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")

# Define our datasets
train_dataset  = TopicDataset(tweets = train['tweet_text'], targets = train['topic'], tokenizer = tokenizer, max_len = 128)
val_dataset  = TopicDataset(tweets = val['tweet_text'], targets = val['topic'], tokenizer = tokenizer, max_len = 128)
test_dataset  = TopicDataset(tweets = train['tweet_text'], targets = test['topic'], tokenizer = tokenizer, max_len = 128)

# Now we define our Pytorch Lightning module
## The main Pytorch Lightning module
class BertModule(pl.LightningModule):

    def __init__(self,
                 num_labels: int = 3,
                 **kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        self._add_model_specific_hparams()
        self._add_default_hparams()
        self._set_seed(self.hparams.random_seed)
        
        self.num_labels = num_labels
        config = AutoConfig.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")
        self.bert = AutoModel.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0", config = config)
        
        self.pre_classifier = torch.nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout = torch.nn.Dropout(self.bert.config.seq_classif_dropout)

        # relu activation function
        self.relu =  torch.nn.ReLU()
        
        self.trainer_params = self._get_trainer_params()

    
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
        

    def training_step(self, batch, batch_nb):
        # batch
        input_ids = batch['input_ids']
        label = batch['label']
        attention_mask = batch['attention_mask']
        # fwd
        y_hat = self(input_ids, attention_mask, label)
        
        # loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(y_hat.view(-1, self.num_labels), label.view(-1))
        
        # logs
        tensorboard_logs = {'train_loss': loss, 'learn_rate': self.optim.param_groups[0]['lr'] }
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
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
        
        # logs
        tensorboard_logs = {'val_loss': loss, 'val_acc': val_acc}
        # can't log in validation step lossess, accuracy.  It wouldn't log it at every validation step
        return {'val_loss': loss, 'val_acc': val_acc, 'progress_bar': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        
        # logs
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_val_acc}
        return {'val_loss': avg_loss, 'progress_bar': tensorboard_logs, 'log': tensorboard_logs}
    
    def on_batch_end(self):
        #for group in self.optim.param_groups:
        #    print('learning rate', group['lr'])
        # This is needed to use the One Cycle learning rate that needs the learning rate to change after every batch
        # Without this, the learning rate will only change after every epoch
        if self.sched is not None:
            self.sched.step()
    
    def on_epoch_end(self):
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
        
        return {'test_loss':loss, 'test_acc': torch.tensor(test_acc)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        tensorboard_logs = {'avg_test_loss': avg_loss, 'avg_test_acc': avg_test_acc}
        return {'avg_test_acc': avg_test_acc, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}
    
    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.hparams.learning_rate, eps=1e-08)
        #scheduler = StepLR(optimizer, step_size=1, gamma=0.2)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-5, epochs = self.hparams.max_epochs, steps_per_epoch = len(train))

        self.sched = scheduler
        self.optim = optimizer
        return [optimizer], [scheduler]

    def train_dataloader(self):
        #dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        #return DataLoader(train_dataset, sampler=dist_sampler, batch_size=32)
        return DataLoader(train_dataset, shuffle = True, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
         return DataLoader(val_dataset,batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(test_dataset,batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

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
    
    def _get_trainer_params(self) -> Dict:

        backup_callback = ModelCheckpoint(
            dirpath=self.hparams.output_path,
            filename="backup-{epoch}-{avg_val_loss:.2f}",
            every_n_epochs=self.hparams.backup_n_epochs,
            save_on_train_epoch_end=True,
            verbose=self.hparams.verbose,
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.hparams.output_path,
            filename="{epoch}-{avg_val_loss:.2f}",
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
            backup_callback,
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

    def load_from_checkpoint(cls, best_checkpoint_path, **kwargs) -> pl.LightningModule:
        return cls.load_from_checkpoint(checkpoint_path=best_checkpoint_path)

    def _add_default_hparams(self) -> None:
        default_params = {
            "random_seed": 42,
            "deterministic": True,
            "shuffle_train_dataset": True,
            "batch_size": 32,
            "loader_workers": 8,
            "output_path": path_to_topic,
            # Trainer params
            "verbose": True,
            "accumulate_grad_batches": 1,
            "accelerator": "auto",
            "devices": 1,
            "max_epochs": 10,
            # Callback params
            "checkpoint_monitor": "avg_val_loss",
            "checkpoint_monitor_mode": "min",
            "early_stop_monitor": "avg_val_loss",
            "early_stop_min_delta": 0,
            "early_stop_patience": 10,
            "backup_n_epochs": 5,
            # Optimizer params
            "optimizer_name": "adamw",
            "optimizer_lr": 1e-04,
            "optimizer_weight_decay": 1e-3,
            "lr_scheduler_factor": 0.2,
            "lr_scheduler_patience": 5,
            "lr_scheduler_min_lr": 1e-7,
        }
        self.hparams.update({**default_params, **self.hparams})

    def _add_model_specific_hparams(self) -> None:
        pass


model = BertModule()
model.fit()
del model