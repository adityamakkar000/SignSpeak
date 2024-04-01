import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelInfo:
  def info(self,layers=False):
    total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    if layers:
      print("Layers:")
      for name, param in self.named_parameters():
          print(f"Layer: {name}, Size: {param.size()}, Parameters: {param.numel()}, Device: {param.device}")


class outputRNN(nn.Module):

  def __init__(self, hidden_size=64, output_size=10, device='cpu', dropout=0.2):
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
