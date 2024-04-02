
# for local machine only
import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("path_to_imports")) #set path to structure/models

# !pip install torcheval # for colab
# !pip install wandb
# !pip install pytorch_lightning

import numpy as np
import random
import pandas
import time
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_f1_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import lightning as L

#local imports
from encoder import Encoder
from LSTM import LSTM
from GRU import GRU
from DS import getDataset

#set determinstic behaviour
torch.use_deterministic_algorithms(True)

# hyper parameters
batch_size = 32
epochs = 1000
learning_rate = 1e-2
time_steps = 10
n_emb = 5
classes=10
padding_value = 0
seed = 1337

#set seeds
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

#read data
data = pandas.read_csv('./data/Data.csv')

#prepare classes
y = data['word'] # extract classes
stoi  = {s:i for i,s in enumerate(sorted(set(y)))}
encode = lambda s: stoi[s]
y = torch.tensor([encode(s) for i,s in enumerate(y)]) # encode letters into words
# y = F.one_hot((y.view(y.shape[0], 1)),10) # don't use one hot with cross entropy loss

#prepare data
x = data.iloc[:,2:time_steps*n_emb + 2].to_numpy()
x = [torch.tensor(i)[~torch.isnan(torch.tensor(i))] for i in x] # remove nan
x = pad_sequence([i for i in x], batch_first=True, padding_value=padding_value) # pad
x = x.view(x.shape[0], -1, n_emb).float() # seperate into B x T x n_emb

# mask_x = (x == padding_value)
# print(mask_x.shape)
print(x.dtype, " ", y.dtype)

# set model params for testing
params = {
  'layers': 2,
  'learning_rate': learning_rate,
  'dense_layer': (True,64)
}

def get_model(t, params):
  model_types = {
    "LSTM": LSTM,
    "GRU": GRU,
    "Encoder": Encoder
  }
  return model_types[t](**params)

# call models and check layers
type_of_model = "GRU"
model = get_model(type_of_model, params)
model.info(layers=True)

# set torch generator and data loader seed
g = torch.Generator(device='cpu').manual_seed(seed)
def seed_worker(worker_id):
  worker_seed = seed
  np.random.seed(worker_seed)
  random.seed(worker_seed)

# create k fold
splits = 5
kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

project_name = "LSTM test with batch loader" # project name
wandb_log = False

# enumerate through the splits with train and test
for i, (train, test) in enumerate(kfold.split(x.cpu(),y.cpu())):

  # get data loaders for the training and test set
  train = DataLoader(getDataset(x[train], y[train]),
                          batch_size=batch_size,
                          shuffle=True,
                          worker_init_fn=seed_worker,
                          generator=g)
  val = DataLoader(getDataset(x[test], y[test]),
                          batch_size=len(test))

  # ensure seed is the same
  torch.manual_seed(seed)
  model = get_model(type_of_model, params) # intialize model

  trainer = L.Trainer(max_epochs=epochs)
  trainer.fit(model, train_dataloaders=train, val_dataloaders=val)
  # accuracy function
