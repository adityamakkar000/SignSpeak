import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import *

class GRU(nn.Module, ModelInfo):

  def __init__(self,input_size=5,hidden_size=64, output_size=10, batch_first=True, layers=1,
               dense_layer=(False, 64), dropout=0.2, device='cpu'):
    super().__init__()
    self.dense, self.dense_size = dense_layer
    self.output_size = output_size
    self.RNN = nn.Sequential(
      nn.Linear(input_size,self.dense_size) if self.dense else None,
      nn.GRU(self.dense_size if self.dense else input_size,
              hidden_size,
              num_layers = layers,
              batch_first =
              batch_first,
              device=device)
    )
    self.output_layers = outputRNN(hidden_size=hidden_size,
                                   output_size=output_size,
                                   device=device,
                                   dropout=dropout)

  def forward(self, x, y_targets):
    hidden_states, outputs = self.RNN(x)
    logits = self.output_layers(outputs[-1,:,:])
    logits = logits.view(-1, self.output_size)
    loss = F.cross_entropy(logits,y_targets)
    return logits, loss
