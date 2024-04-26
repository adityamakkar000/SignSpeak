
import numpy as np
import torch
import torch.nn as nn
import pandas
import torch.nn.functional as F
import time
from sklearn.model_selection import StratifiedKFold
from torcheval.metrics.functional import multiclass_f1_score

# hyper parameters
classes = 10
n_emb = 5
n_layers = 1
n_heads = 5
n_layers = 3
time_steps = 10
batch_size = 32
epochs = 300000
learning_rate = 1e-4
dropout = 0.2

#encoder layer
data = pandas.read_csv('./Data.csv')
#prepare classes
y = data['word']
stoi  = {s:i for i,s in enumerate(sorted(set(y)))}
encode = lambda s: stoi[s]
y = torch.tensor([encode(s) for i,s in enumerate(y)])
# y = F.one_hot((y.view(y.shape[0], 1)),10) # don't use one hot with cross entropy loss

#prepare data
x = torch.tensor(data.iloc[:,2:time_steps*n_emb + 2].to_numpy())
x = torch.nan_to_num(x).float()
x = x.view(x.shape[0], -1, n_emb)
print(x.dtype, " ", y.dtype)
print(x.shape)

class LSTMStacked(nn.Module):

  def __init__(self, batch_first=True, layers=1, dense_layer=False):
    super().__init__()
    self.RNN = nn.LSTM(5, 64, num_layers = layers, batch_first = batch_first)
    self.output_layers = nn.Sequential(
        nn.Linear(64,64),
        nn.Dropout(0.2),
        nn.Tanh(),
        nn.Linear(64,10))

  def forward(self, x, y_targets):
    hidden_states, (outputs,cell_states) = self.RNN(x)
    logits = self.output_layers(outputs[-1,:,:])
    # print(logits.shape, " ", y_targets.shape)
    logits = logits.view(-1, 10)
    loss = F.cross_entropy(logits,y_targets)
    return logits, loss

class LSTM(nn.Module):

  def __init__(self,input_size=5,hidden_size=64, output_size=10, batch_first=True, layers=1,
               dense_layer=False, dropout=0.2):
    super().__init__()
    self.RNN = nn.LSTM(input_size, hidden_size, num_layers = layers, batch_first = batch_first)
    self.output_layers = nn.Sequential(
        nn.Linear(hidden_size,hidden_size),
        nn.Dropout(dropout),
        nn.Tanh(),
        nn.Linear(hidden_size,output_size)
    )

  def forward(self, x, y_targets):
    hidden_states, (outputs, cell_states) = self.RNN(x)
    logits = self.output_layers(outputs[-1,:,:])
    # print(logits.shape, " ", y_targets.shape)
    logits = logits.view(-1, 10)
    loss = F.cross_entropy(logits,y_targets)
    return logits, loss


model = LSTMStacked(batch_first=True, layers=2)
print(sum([p.nelement() for p in model.parameters()]))

splits = 5
kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=1337)

for i, (train, test) in enumerate(kfold.split(x,y)):
  Xtr,Ytr = x[train], y[train]
  Xval, Yval = x[test], y[test]
  model = LSTMStacked(batch_first=True, layers=2)

  def get_batches(split):
    x_values, y_values = {
        'train': [Xtr, Ytr],
        'test': [Xval, Yval]
    }[split]
    idx = torch.randint(0, x_values.shape[0], (batch_size,))
    return x_values[idx], y_values[idx]

  @torch.no_grad()
  def get_val_loss(x_val, y_val):
    model.eval()
    logits, val_loss = model(x_val, y_val)
    # val_f1 = multiclass_f1_score(logits, y_val,num_classes=classes, average=None)
    model.train()
    return val_loss.item()

  optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
  start_time = time.time()
  for _ in range(epochs):
    x_epoch, y_epoch = get_batches('train')
    logits, loss = model(x_epoch, y_epoch)
    f1 = multiclass_f1_score(logits,y_epoch, num_classes=classes, average=None)
    f1_average = f1.mean()
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
    if(_ % 1000 == 0):
      end_time = time.time()
      delta_time = end_time - start_time
      val_loss = get_val_loss(Xval, Yval)
      # val_loss, val_f1 = get_val_loss(Xval, Yval)
      print("epoch", _,
            "   Training Loss", loss.item(),
            "    Training F-1", f1_average,
            "     Val loss", val_loss,
            # "    Val F-1", val_f1.mean(),
            "     Time", delta_time)

      start_time = time.time()
