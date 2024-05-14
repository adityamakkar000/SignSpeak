
# for local machines
import sys
import argparse
import os
import os.path as op

path = op.dirname(op.dirname(op.dirname(op.realpath(__file__))))
sys.path.append(path)

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
from src.models.encoder import Encoder
from src.models.LSTM import LSTM
from src.models.GRU import GRU
from src.misc.DataModule import ASLDataModule

#set determinstic behaviour
# torch.use_deterministic_algorithms(True) # don't use on GPU

# setup cli arg parser
parser = argparse.ArgumentParser(description="model traning")
parser.add_argument('-description', dest='description', type=str, required=False) # description for wandb

# general params
parser.add_argument('-time_steps', dest='time_steps', type=int, required=False) # time steps
parser.add_argument('-lr', dest='lr', type=float, required=False) # learning rate
parser.add_argument('-batch_size', dest='batch_size', type=int, required=False) # batch size
parser.add_argument('-epochs', dest='epochs', type=int, required=False) # epochs

parser.add_argument('-model', dest='model', type=str, required=False) # get type of model
parser.add_argument('-layers', dest='layers', type=int, required=False) # number of layers
parser.add_argument('-hidden_size', dest='hidden_size', type=int, required=False) # hidden size

# RNN params
parser.add_argument('-dense_layer', dest='dense_layer', type=bool, required=False) # dense layer
parser.add_argument('-dense_size', dest='dense_size', type=int, required=False) # dense size


# encoder params
parser.add_argument('-number_heads', dest='number_heads', type=int, required=False) # number of heads


args = parser.parse_args()


# hyper parameters
batch_size = args.batch_size if args.batch_size else 64
epochs = args.epochs if args.epochs else  600
learning_rate = args.lr if args.lr else 1e-4
time_steps = args.time_steps if args.time_steps else 60
n_emb = 5
classes=36
seed = 1337


#set seeds
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# set model params for testing
encoder_params = {
  'layers': args.layers if args.layers else 1,
  'time_steps': time_steps,
  'number_heads': args.number_heads if args.number_heads else 1,
  'input_size': 5,
  'classes': classes,
  'hidden_size': args.hidden_size if args.hidden_size else 5,
  'learning_rate': learning_rate,
}

RNN_params = {
  'input_size': 5,
  'layers': args.layers if args.layers else 1,
  'dense_layer': (True, args.dense_size) if args.dense_layer else (False, 64),
  'hidden_size': args.hidden_size if args.hidden_size else 64,
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
type_of_model = args.model if args.model else "Encoder"
model = get_model(type_of_model, params)
model.info(layers=True)
print(model.total_params())


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
            "total parameters": model.total_params(),
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
