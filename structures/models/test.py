"""
Script to test out model
Writes out all layers without calling any training/data processing programs
"""

from encoder import Encoder
from LSTM import LSTM
from GRU import GRU

model = GRU(dense_layer=(True,64))
model.info(layers=True)
