import torch
import torch.nn as nn
import torch.nn.functional as F


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
