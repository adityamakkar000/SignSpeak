import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.data import Dataset


class getDataset(Dataset):

  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, idx):

    sample = {
      "measurement": self.x[idx],
      "label": self.y[idx]
    }

    return sample
