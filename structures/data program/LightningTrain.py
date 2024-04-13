# for local machines
import sys
import argparse
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("path_to_imports")) #set path to structure/models

# !pip install torcheval # for colab
# !pip install wandb
# !pip install pytorch_lightning

import numpy as np
import random
import time
import wandb
import torch
import lightning as L
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
parser.add_argument('-description', dest='description', type=str, required=True)
args = parser.parse_args()

# hyper parameters
batch_size = 32
epochs = 1000
learning_rate = 1e-2
time_steps = 10
n_emb = 5
classes=10
seed = 1337

#set seeds
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# set model params for testing
params = {
  # 'layers': 2,
  'learning_rate': learning_rate,
  # 'dense_layer': (True,64)
}

def get_model(t, params):
  """ take model type and return that with the desired parameters"""
  model_types = {
    "LSTM": LSTM,
    "GRU": GRU,
    "Encoder": Encoder
  }
  return model_types[t](**params)

# call models and check layers
type_of_model = "Encoder"
model = get_model(type_of_model, params)
model.info(layers=True)


splits = 5 #k-fold splits
project_name="SignSpeak"
wandb_log = False

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
            "kfold-split": i+1
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
    'shuffle': True,
    'generator': g
  }

  # intialize dataset, traniner and fit model
  dataset = ASLDataModule(**dataset_params)
  trainer = L.Trainer(**trainer_params, logger=wandb_logger) if wandb_log else L.Trainer(**trainer_params)
  trainer.fit(model, dataset)
