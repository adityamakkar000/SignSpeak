# for local machines
import sys
import argparse
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("path_to_imports")) #set path to structure/models

# !pip install torcheval # for colab
# !pip install wandb
# !pip install Lightning

import numpy as np
import random
import time
import wandb
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from datetime import datetime

#local imports
from encoder import Encoder
from LSTM import LSTM
from GRU import GRU
from DS import ASLDataModule

#set determinstic behaviour
torch.use_deterministic_algorithms(True) # don't use on GPU

# setup cli arg parser
parser = argparse.ArgumentParser(description="get character and word count")
parser.add_argument('-description', dest='description', type=str, required=False)
# add argument for model type and params
parser.add_argument('-model', dest='model', type=str, required=False)

args = parser.parse_args()


# hyper parameters
batch_size = 64
epochs = 600
learning_rate = 1e-4
time_steps = 60
n_emb = 5
classes=36
seed = 1337

#set seeds
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# set model params for testing
encoder_params = {
  'layers': 2,
  'time_steps': time_steps,
  'number_heads': 1,
  'input_size': n_emb,
  'classes': classes,
  'hidden_size': n_emb,
  'learning_rate': learning_rate,
}

RNN_params = {
  'input_size': n_emb,
  'layers': 2,
  'dense_layer': (False, 64),
  'hidden_size': 64,
  'classes': classes,
  'learning_rate': learning_rate,
}

params = {
          "Encoder": encoder_params,
          "LSTM": RNN_params,
          "GRU": RNN_params
}

def get_model(t, params):
  """ take model type and return that with the desired parameters"""
  model_types = {
    "LSTM": LSTM,
    "GRU": GRU,
    "Encoder": Encoder
  }
  return model_types[t](**params[t])

# call models and check layers
type_of_model = "Encoder"
model = get_model(type_of_model, params)
model.info(layers=True)


splits = 5 #k-fold splits
project_name="SignSpeak"
wandb_log = True if args.description else False
print(wandb_log, " ", args)

# k-fold number
for split_number in range(splits):

  # ensure seed is the same
  torch.manual_seed(seed)
  model = get_model(type_of_model, params) # intialize model

  if wandb_log:
    run_name = type_of_model + "_" + args.description + "_" + str(split_number+1) + "-fold_" + datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    wandb_logger = WandbLogger(project=project_name, name=run_name) # setup wandb logger
    config={
            "learning_rate": learning_rate,
            "context length": time_steps,
            "params": params,
            "classes": classes,
            "seed": seed,
            "epochs": epochs,
            "kfold-split": split_number+1
    }
    wandb_logger.experiment.config.update(config)

  trainer_params = {
    "max_epochs": epochs,
    "log_every_n_steps": 1
  }

  dataset_params = {
    'n_emb': n_emb,
    'time_steps': time_steps,
    'kfold': split_number,
    'splits': splits,
    'seed': seed,
    'batch_size': batch_size,
    'shuffle': True
  }

  # intialize dataset, traniner and fit model
  dataset = ASLDataModule(**dataset_params)
  trainer = L.Trainer(**trainer_params, logger=wandb_logger) if wandb_log else L.Trainer(**trainer_params)
  trainer.fit(model, dataset)

  if wandb_log:
    wandb.finish()
