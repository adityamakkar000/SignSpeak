
# implmention of encoder only transformer from attention is all you need



import numpy as np
import torch
import torch.nn as nn
import pandas
import torch.nn.functional as F


#hyper paramters

vocab_size = 10 # measurment from fingers
n_emb = 5 # how many I want to transform too
time_steps = 10 # time steps
batch_size = 32 # how many batches to use
n_heads = 4

#encoder layer
data = pandas.read_csv('./data/Data.csv')
#prepare classes
y = data['word']
stoi  = {s:i for i,s in enumerate(sorted(set(y)))}
encode = lambda s: stoi[s]
y = torch.tensor([encode(s) for i,s in enumerate(y)])
# y = F.one_hot((y.view(y.shape[0], 1)),10) # don't use one hot with cross entropy loss

#prepare data
x = torch.tensor(data.iloc[:,2:time_steps*n_emb + 2].to_numpy())
x = torch.nan_to_num(x)
x = x.view(x.shape[0], -1, n_emb)

class FFN(nn.Module):

  def __init__(self, n_emb):
    super().__init__()
    self.ffnl = nn.Sequential(
      nn.Linear(n_emb, n_emb*4),
      nn.ReLU(),
      nn.Linear(n_emb*4, n_emb)
    )

  def forward(self, x):
    self.out = self.ffnl(x)

class Head(nn.Module):

  def __init__ (self, n_emb, head_size):
    super().__init__()
    self.key = nn.Linear(n_emb, head_size, bias=False)
    self.queries = nn.Linear(n_emb, head_size, bias=False)
    self.values = nn.Linear(n_emb, head_size, bias=False)

  def forward(self, x):

    self.k = self.key(x)
    self.q = self.queries(x)
    self.v = self.values(x)

    self.weighting = (self.q @ torch.transpose(self.k, -1,-2))*(n_emv**-0.5)

    self.out = self.weighting @ self.v

    return self.out

class multiLayerAttention(nn.Module):

  def __init__(self, n_emb, n_heads):
    super().__init__()
    self.head_size = n_emb // n_heads
    self.heads =  nn.ModuleList([Head(n_emb, self.head_size) for i in range(n_heads)])

  def forward(self, x):

    self.out = torch.concatenate([h(x) for h in self.heads], dim=-1)

class block(nn.Module):

  def __init__(self, n_emb, n_heads):
    super().__init__()
    self.self_attention = multiLayerAttention(n_emb, n_heads)
    self.layer1_norm = nn.LayerNorm(n_emb)
    self.layer2_norm = nn.LayerNorm(n_emb)
    self.ffnn = FFN(n_emb)

  def forward(self, x):
    self.comm = x + self.self_attention(self.layer1_norm(x))
    self.out = x + self.ffnn(self.layer1_norm(self.comm))


class transformer(nn.Module):

  def __init__(self, layers, n_heads=4):
    super().__init__()
    self.embedding_table = nn.Embedding(vocab_size, n_emb)
    self.pos_embedding_table = nn.Embedding(time_steps, n_emb)
    self.blocks = nn.Sequential(*[block(n_emb, n_heads) for _ in range(layers)])
    self.final_norm = nn.LayerNorm(n_emb)
    self.linear = nn.Linear(n_emb, vocab_size)


  def forward(self, x, y_targets):
    self.x = self.embedding_table(x)
    self.pos = self.pos_embedding_table(torch.arange(time_steps))
    self.x = self.x + self.pos
    self.x = self.blocks(self.x)
    logits = self.linear(self.final_norm(self.x))

    logits = logits.view(-1,vocab_size)
    y_targets = y_targets.view(-1)
    loss = F.cross_entropy(logits, y_targets)

    return logits, loss


model = transformer(4)
optimizer = torch.optim.AdamW(model.parameters(),lr=4e-3)
epochs = 1000

def get_batches():
  n = int(0.9*x.shape[0])
  Xtr = x[:n,:,:]
  Ytr = y[:n,:]

  Xval = x[n:,:,:]
  Yval = y[n:]

  return Xtr, Ytr, Xval, Yval

Xtr, Ytr, Xval, Yval = get_batches()




@torch.no_grad()
def get_loss(tr_loss):
  idx = torch.randint(0, Xval.shape[0], (batch_size,))
  Xb, Yb = Xval[idx], Yval[idx]
  x_out, val_loss = model(Xb, Yb)

  return tr_loss, val_loss

for _ in range(epochs):

  idx = torch.randn(0, Xtr.shape[0], (batch_size,))
  Xb = Xtr[idx]
  Yb = Ytr[idx]
  x_predictions, loss = model(Xb, Yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
  if(_ % 100 == 0):
    print(loss.item())
