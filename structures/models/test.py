import torch
import torch.nn
import torch.nn.functional as F
from encoder import Encoder
from LSTM import LSTM
from GRU import GRU



model = GRU(dense_layer=(True,64))
model.info(layers=True)
