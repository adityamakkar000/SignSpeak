import tensorflow as tf
from tensorflow.keras import layers, models
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pymongo
import os
from dotenv import load_dotenv
import keras
import random

load_dotenv()
client = pymongo.MongoClient(os.getenv("MONGO_URI"))
db = client['RNN_9_Newaccuracy_64neurons_withCMs']
collection = [
    db[('SIMPLE_1_layer_nodense')],
    db[('SIMPLE_2_layer_nodense')],
    db[('GRU_1_layer_nodense')],
    db[('GRU_2_layer_nodense')],
    db[('LSTM_1_layer_nodense')],
    db[('LSTM_2_layer_nodense')],
    db[('SIMPLE_1_layer_dense')],
    db[('SIMPLE_2_layer_dense')],
    db[('GRU_1_layer_dense')],
    db[('GRU_2_layer_dense')],
    db[('LSTM_1_layer_dense')],
    db[('LSTM_2_layer_dense')]
]

for col in collection:
  arr = col.find()
  print(len(arr))
  for i in arr:
    cm_avg = np.zeros((10,10))
    for cm in i['confusion_matrix']:
      cm_avg = cm_avg + cm
    cm_avg = cm_avg / 5
    print(cm_avg)

