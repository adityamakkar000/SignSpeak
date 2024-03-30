import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):

  def __init__(self, input_size, head_size, dropout, linearBias=False):
    super().__init__()
    self.head_size = head_size
    self.keys = nn.Linear(input_size, head_size, bias=linearBias)
    self.queries = nn.Linear(input_size, head_size, bias=linearBias)
    self.values = nn.Linear(input_size, head_size, bias=linearBias)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    k = self.keys(x)
    q = self.queries(x)
    v = self.values(x)

    weights = (q @ torch.transpose(k, 1,2))*(self.head_size)**-0.5
    weights = F.softmax(weights, dim=1)
    weights = self.dropout(weights)
    logits = weights @ v
    # print("logits shape", logits.shape)
    return logits


class MultiHeadAttention(nn.Module):

  def __init__(self,input_size, head_size, dropout):
    super().__init__()
    self.heads = nn.ModuleList([Head(input_size, head_size, dropout).cuda() for i in range(n_heads)])
    self.layer = nn.Sequential(
        nn.Linear(input_size, input_size),
        nn.Dropout(dropout)
    )

  def forward(self, x):
    multi_head_attention = [head(x) for head in self.heads]
    # print("multi_head_attention shape  ", multi_head_attention)
    logits = torch.cat(multi_head_attention, -1)
    logits = self.layer(logits)
    return logits

class FeedForwardNetwork(nn.Module):

  def __init__(self, input_size, dropout):
    super().__init__()
    self.feed_forward = nn.Sequential(
        nn.Linear(input_size, input_size*4),
        nn.ReLU(),
        nn.Linear(input_size*4, input_size),
        nn.Dropout(dropout)
    )

  def forward(self, x):
    logits = self.feed_forward(x)
    return logits

class Block(nn.Module):

  def __init__(self, input_size, dropout, head_size):
    super().__init__()
    self.heads = MultiHeadAttention(input_size,head_size,dropout)
    self.feed_forward = FeedForwardNetwork(input_size, dropout)
    self.layer1_norm = nn.LayerNorm(input_size)
    self.layer2_norm = nn.LayerNorm(input_size)

  def forward(self, x):
    x = x + self.heads(self.layer1_norm(x))
    logits = x + self.feed_forward(self.layer2_norm(x))

    return logits

class Encoder(nn.Module):

  def __init__(self,layers=1, input_size=10, hidden_size=5,time_steps=10, dropout=0.2):
    super().__init__()
    head_size = hidden_size // layers
    self.time_steps = time_steps
    self.dim = input_size
    self.embedding_table = nn.Embedding(input_size, hidden_size)
    self.pos_embedding_table = nn.Embedding(time_steps, hidden_size)
    self.blocks = nn.Sequential(*[Block(hidden_size,dropout, head_size).cuda()for i in range(layers)])
    self.final_layer_norm = nn.LayerNorm(hidden_size)
    self.linear_output = nn.Linear(hidden_size, input_size)

  def forward(self, x_input, y_targets):

    pos = self.pos_embedding_table(torch.arange(self.time_steps).to('cuda'))
    x = x_input + pos
    x = self.blocks(x)
    x = self.final_layer_norm(x)
    x = self.linear_output(x)
    logits = x[:,-1,:].view(x.shape[0],1,input_size)
    B,T,C = logits.shape
    logits = logits.view(B*T, C)
    y_targets = y_targets.view(B*T).long()

    loss = F.cross_entropy(logits, y_targets)
    return logits, loss
