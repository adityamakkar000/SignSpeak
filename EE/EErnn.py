import tensorflow as tf
from tensorflow.keras import layers, models
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
import pymongo
from model import researchModel
import os
from dotenv import load_dotenv

load_dotenv()

client = pymongo.MongoClient(os.getenv("MONGO_URI"))
db = client['cluster0']
collection = [
    db[('SIMPLE_RNN_1_layer')],
    db[('SIMPLE_RNN_2_layer')],
    db[('GRU_RNN_1_layer')],
    db[('GRU_RNN_2_layer')],
    db[('LSTM_RNN_1_layer')],
    db[('LSTM_RNN_2_layer')]
]

mapping = {'a': [1,0,0,0,0,0,0,0,0],
        'b': [0,1,0,0,0,0,0,0,0],
        'c': [0,0,1,0,0,0,0,0,0],
        'd': [0,0,0,1,0,0,0,0,0],
        'e': [0,0,0,0,1,0,0,0,0],
        'f': [0,0,0,0,0,1,0,0,0],
        'g': [0,0,0,0,0,0,1,0,0],
        'h': [0,0,0,0,0,0,0,1,0],
        'i': [0,0,0,0,0,0,0,0,1]}

data = pandas.read_csv('./EEdata.csv')

words = data['word']
y = data['word']
x = data.iloc[:,2:]
data.iloc[:,2:]
y = np.concatenate([[mapping[i] for i in y]])
x = x.to_numpy()
x = x[:,:45].reshape(900,5,9)
x = x.transpose(0, 2, 1)
x = np.nan_to_num(x)

print(x.shape)

batch_size = 64
epochs = 1000
trials = 25

# for i in range(trials):
#     m1 = researchModel("SimpleRNN", 1, batch_size, epochs, x, y)
#     results1 = m1.evaluate(x, y)
#     collection[0].insert_one({'model': 'SIMPLE_RNN_1_layer', 'loss': results1[0], 'accuracy': results1[1]})

# for i in range(trials):
#     m2 = researchModel("SimpleRNN", 2, batch_size, epochs, x, y)
#     results2 = m2.evaluate(x, y)
#     collection[1].insert_one({'model': 'SIMPLE_RNN_2_layer', 'loss': results2[0], 'accuracy': results2[1]})

# for i in range(trials):
#     m3 = researchModel("GRU", 1, batch_size, epochs, x, y)
#     results3 = m3.evaluate(x, y)
#     collection[2].insert_one({'model': 'GRU_RNN_1_layer', 'loss': results3[0], 'accuracy': results3[1]})

for i in range(trials):
    m4 = researchModel("GRU", 2, batch_size, epochs, x, y)
    results4 = m4.evaluate(x, y)
    collection[3].insert_one({'model': 'GRU_RNN_2_layer', 'loss': results4[0], 'accuracy': results4[1]})

for i in range(trials):
    m5 = researchModel("LSTM", 1, batch_size, epochs, x, y)
    results5 = m5.evaluate(x, y)
    collection[4].insert_one({'model': 'LSTM_RNN_1_layer', 'loss': results5[0], 'accuracy': results5[1]})

for i in range(trials):
    m6 = researchModel("LSTM", 2, batch_size, epochs, x, y)
    results6 = m6.evaluate(x, y)
    collection[5].insert_one({'model': 'LSTM_RNN_2_layer', 'loss': results6[0], 'accuracy': results6[1]})


