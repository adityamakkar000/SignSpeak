
# for local machine only
import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("p")) #set path to structure/models

# !pip install torcheval # for colab

import numpy as np
import torch
import torch.nn as nn
import pandas
import torch.nn.functional as F
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from torcheval.metrics.functional import multiclass_f1_score
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.data import Dataset, DataLoader
import wandb
from encoder import Encoder
from LSTM import LSTM
from GRU import GRU
from DS import getDataset 

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
a
data = pandas.read_csv('./data/Data.csv')

#prepare classes
y = data['word']
stoi  = {s:i for i,s in enumerate(sorted(set(y)))}
encode = lambda s: stoi[s]
y = torch.tensor([encode(s) for i,s in enumerate(y)])
# y = F.one_hot((y.view(y.shape[0], 1)),10) # don't use one hot with cross entropy loss

#prepare data
x = data.iloc[:,2:time_steps*n_emb + 2].to_numpy()
x = [torch.tensor(i)[~torch.isnan(torch.tensor(i))] for i in x]
x = pad_sequence([i for i in x], batch_first=True, padding_value=padding_value)
x = x.view(x.shape[0], -1, n_emb).float()

# mask_x = (x == padding_value)
# print(mask_x.shape)
x,y = x.to(device), y.to(device)
print(x.device, y.device)
print(x.dtype, " ", y.dtype)

params = {
  'layers': 10,
  'number_heads': 1,
  'input_size': 10,
  'hidden_size': 5,
  'time_steps': 10,
  'dropout': 0.2,
  'device': device
}

model = LSTM()
model.info(layers=True)


g = torch.Generator(device='cpu').manual_seed(seed)
splits = 5
kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
project_name = "LSTM test with batch loader"


for i, (train, test) in enumerate(kfold.split(x.cpu(),y.cpu())):
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

  train = DataLoader(getDataset(x[train], y[train]),
                          batch_size=batch_size,
                          shuffle=True,
                          generator=g)
  val = DataLoader(getDataset(x[test], y[test]),
                          batch_size=len(test),
                          shuffle=False,
                          generator=g)

  torch.manual_seed = seed
  model = LSTM(**params)

  def get_accuracy(cm):
    TP = np.diag(cm)
    TN = np.sum(np.sum(cm,axis=0))-(np.sum(cm, axis = 0) + np.sum(cm, axis=1) - np.diag(cm))
    FP = np.sum(cm, axis = 0) - np.diag(cm)
    FN = np.sum(cm, axis=1) - np.diag(cm)
    true_acc = (TP + TN)/(TP + TN + FP + FN)
    cat_acc = TP/(FN+TP)
    return true_acc, cat_acc

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

  optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
  start_time = time.time()
  for _ in range(epochs):

    train_loss_epoch = 0
    for idx, sample in enumerate(train):
      # prepare batches from data loader
      m, l = sample
      x_batch, y_batch = sample[m], sample[l]
      logits, loss = model(x_batch, y_batch)

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
      wandb.log(data)

    train_loss_epoch /= batch_size
    end_time = time.time()
    delta_time = end_time - start_time
    print(f"epoch {_:<4}   Training Loss:{train_loss_epoch:.4f}   Val-loss:{val_loss:.4f}   Val-F-1:{val_f1.mean().item():.4f} Categorical-Accuracy:{torch.tensor(categorical_accuracy).mean().item():.4f}  True-Accuracy:{torch.tensor(accuracy).mean().item():.4f}   Time {delta_time:.1f}")
    start_time = time.time()
  wandb.finish()
