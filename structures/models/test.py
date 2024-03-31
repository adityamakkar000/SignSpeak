import torch
import torch.nn
import torch.nn.functional as F
from encoder import Encoder
from LSTM import LSTM
from GRU import GRU



model = LSTM()
model.info(layers=True)
