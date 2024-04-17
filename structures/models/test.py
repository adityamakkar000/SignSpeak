"""
Script to test out model
Writes out all layers without calling any training/data processing programs
"""

# from encoder import Encoder
# from LSTM import LSTM
# from GRU import GRU

# model = GRU(dense_layer=(True,64))
# model.info(layers=True)


import torch
import pandas


data = pandas.read_csv("./data/test2.csv", dtype={"word": "string"})
y = data['word'] # extract classes
number = [str(i) for i in range(1,11)]
indexes = [i for i,s in enumerate(y) if s not in number] # remove numbers
y = [s for i,s in enumerate(y) if s not in number]
stoi  = {s:i for i,s in enumerate(sorted(set(y)))} # assign indexes to each possible class (a - z, 1-10)
encode = lambda s: stoi[s] # inline function to covert character to class number
y = torch.tensor([encode(s) for i,s in enumerate(y)]) # encode letters into words

print(y.shape)
