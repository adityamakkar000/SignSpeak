import torch
import torch.nn as nn
import numpy as np

class ModelInfo:

  def get_accuracy(self, cm):
    TP = np.diag(cm)
    TN = np.sum(np.sum(cm,axis=0))-(np.sum(cm, axis = 0) + np.sum(cm, axis=1) - np.diag(cm))
    FP = np.sum(cm, axis = 0) - np.diag(cm)
    FN = np.sum(cm, axis=1) - np.diag(cm)
    true_acc = (TP + TN)/(TP + TN + FP + FN)
    cat_acc = TP/(FN+TP)
    return true_acc, cat_acc

  def info(self,layers=False):
    total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    if layers:
      print("Layers:")
      for name, param in self.named_parameters():
          print(f"Layer: {name}, Size: {param.size()}, Parameters: {param.numel()}, Device: {param.device}")

class outputRNN(nn.Module):

  def __init__(self, hidden_size=64, transformed_size=32, output_size=10, dropout=0.2):
    super().__init__()
    self.output_layers = nn.Sequential(
          nn.Linear(hidden_size,transformed_size),
          nn.Dropout(dropout),
          nn.Tanh(),
          nn.Linear(transformed_size,output_size)
        )

  def forward(self, x):
    logits = self.output_layers(x)
    return logits
