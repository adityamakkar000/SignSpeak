
# for local machine only
import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("p")) #set path to structure/models

# !pip install torcheval # for colab

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
from torch.nn.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

#local imports
from encoder import Encoder
from LSTM import LSTM
from GRU import GRU
from DS import getDataset

#set determinstic behaviour
torch.use_deterministic_algorithms(True)

#cuda setup
if torch.cuda.is_available():
  print("gpu usage:", torch.cuda.is_available())
  print("current device", torch.cuda.current_device())
  print("gpu name:",torch.cuda.get_device_name(0))
  print("memory allocated:",torch.cuda.memory_allocated())
  print("memory reserved:", torch.cuda.memory_reserved())
  device = 'cuda'
else:
  device = 'cpu'

# hyper parameters
batch_size = 32
epochs = 1000
learning_rate = 1e-4
time_steps = 10
n_emb = 5
classes=10
device = device
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
x,y = x.to(device), y.to(device) # move to device
print(x.device, y.device)
print(x.dtype, " ", y.dtype)


# set model params for testing
params = {
  'layers': 10,
  'number_heads': 1,
  'input_size': 10,
  'hidden_size': 5,
  'time_steps': 10,
  'dropout': 0.2,
  'device': device
}

# call models and check layers
model = LSTM()
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

# enumerate through the splits with train and test
for i, (train, test) in enumerate(kfold.split(x.cpu(),y.cpu())):

  # setup wandb recording
  wandb.init(
      project=project_name,
      config={
          "learning_rate": learning_rate,
          "context length": time_steps,
          "params": params,
          "classes": classes,
          "seed": seed,
          "epochs": epochs,
          "kfold-split": i+1
      }
    )

  # get data loaders for the training and test set
  train = DataLoader(getDataset(x[train], y[train]),
                          batch_size=batch_size,
                          shuffle=True,
                          worker_init_fn=seed_worker,
                          generator=g)
  val = DataLoader(getDataset(x[test], y[test]),
                          batch_size=len(test),
                          shuffle=False,
                          worker_init_fn=seed_worker,
                          generator=g)

  # ensure seed is the same
  torch.manual_seed(seed)
  model = LSTM(**params) # intialize model

  # accuracy function
  def get_accuracy(cm):
    TP = np.diag(cm)
    TN = np.sum(np.sum(cm,axis=0))-(np.sum(cm, axis = 0) + np.sum(cm, axis=1) - np.diag(cm))
    FP = np.sum(cm, axis = 0) - np.diag(cm)
    FN = np.sum(cm, axis=1) - np.diag(cm)
    true_acc = (TP + TN)/(TP + TN + FP + FN)
    cat_acc = TP/(FN+TP)
    return true_acc, cat_acc

  # val data set forward pass
  @torch.no_grad()
  def get_val_stats(val):
    model.eval()
    for idx, sample in enumerate(val):
      l,m = sample
      x_val, y_val = sample[l], sample[m]
      logits, val_loss = model(x_val, y_val)
      cm = confusion_matrix(y_val.cpu(),
                            logits.cpu().argmax(axis=1).numpy(), labels=np.arange(10).tolist())
      val_f1 = multiclass_f1_score(logits, y_val,num_classes=classes, average=None)
      true_accuracy, categorical_accuracy = get_accuracy(cm)
    model.train()
    return val_loss.item(), val_f1, cm, true_accuracy, categorical_accuracy

  optim = torch.optim.AdamW(model.parameters(), lr=learning_rate) # set optimizer
  start_time = time.time() # for time measurment purposes on epoch
  for _ in range(epochs):

    train_loss_epoch = 0 # for average on one epoch
    for idx, sample in enumerate(train):
      # prepare batches from data loader
      m, l = sample # keys
      x_batch, y_batch = sample[m], sample[l]
      logits, loss = model(x_batch, y_batch) # forward pass

      # train step
      optim.zero_grad(set_to_none=True)
      loss.backward()
      optim.step()

      train_loss = loss.item()
      train_loss_epoch += train_loss
      val_loss, val_f1, cm, accuracy, categorical_accuracy = get_val_stats(val)
      data = {
          "train_loss": train_loss,
          "val_loss": val_loss,
          "micro_F1": val_f1.cpu().numpy().tolist(),
          "macro_F1": val_f1.mean().item(),
          "cat_accuracy": torch.tensor(categorical_accuracy).mean().item(),
          "true_accuracy": torch.tensor(accuracy).mean().item()
      }
      wandb.log(data) # upload data 

    train_loss_epoch /= batch_size
    end_time = time.time()
    delta_time = end_time - start_time
    print(f"epoch {_:<4}   Training Loss:{train_loss_epoch:.4f}   Val-loss:{val_loss:.4f}   Val-F-1:{val_f1.mean().item():.4f} Categorical-Accuracy:{torch.tensor(categorical_accuracy).mean().item():.4f}  True-Accuracy:{torch.tensor(accuracy).mean().item():.4f}   Time {delta_time:.1f}")
    start_time = time.time()
  wandb.finish()
