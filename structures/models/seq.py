import torch
import torch.nn as nn
import torch.nn.functional as F

class outputRNN(nn.Module):

  def __init__(self, hidden_size=64, output_size=10, device='cpu'):
    super().__init__()
    self.output_layers = nn.Sequential(
          nn.Linear(hidden_size,hidden_size, device=device),
          nn.Dropout(dropout),
          nn.Tanh(),
          nn.Linear(hidden_size,output_size, device=device)
        )
    
  def forward(self, x):
    logits = self.output_layers(x)
    return logits
