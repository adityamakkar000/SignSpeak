import torch
import torch.nn
import torch.nn.functional as F
from encoder import Encoder


params = {
  'layers': 1,
  'number_heads': 1,
  'input_size': 10,
  'hidden_size': 5,
  'time_steps': 10,
  'dropout': 0.2
}
model = Encoder(**params)
model.info(layers=True)
