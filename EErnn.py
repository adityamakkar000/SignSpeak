import tensorflow as tf
from tensorflow.keras import layers, models
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
import pymongo
import os
from dotenv import load_dotenv
import keras
import random

def average(lst):
    return sum(lst) / len(lst)

class researchModel:

  def __init__(self, type, stacks, batch_size, epochs, x, y, seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)

    if(stacks == 1):
      if(type == "SimpleRNN"):
        self.encoder = models.Sequential([
          layers.Dense(32, activation='tanh', input_shape=(18,5)),
          layers.SimpleRNN(64,activation='tanh'),
          layers.Dense(32, activation='tanh', input_shape=(32,)),
          layers.Dense(9, activation='softmax')
        ])
      elif(type == "GRU"):
        self.encoder = models.Sequential([
          layers.Dense(32, activation='tanh', input_shape=(18,5)),
          layers.GRU(64,activation='tanh'),
          layers.Dense(32, activation='tanh', input_shape=(32,)),
          layers.Dense(9, activation='softmax')
        ])
      elif(type == "LSTM"):
        self.encoder = models.Sequential([
          layers.Dense(32, activation='tanh', input_shape=(18,5)),
          layers.LSTM(16,activation='tanh'),
          layers.Dense(32, activation='tanh', input_shape=(32,)),
          layers.Dense(9, activation='softmax')
        ])
    elif(stacks == 2):
      if(type == "SimpleRNN"):
        self.encoder = models.Sequential([
          layers.Dense(32, activation='tanh', input_shape=(18,5)),
          layers.SimpleRNN(64,activation='tanh', return_sequences=True),
          layers.SimpleRNN(64,activation='tanh'),
          layers.Dense(32, activation='tanh', input_shape=(64,)),
          layers.Dense(9, activation='softmax')
        ])
      elif(type == "GRU"):
        self.encoder = models.Sequential([
          layers.Dense(64, activation='tanh', input_shape=(18,5)),
          layers.GRU(128,activation='tanh', return_sequences=True),
          layers.GRU(128,activation='tanh'),
          layers.Dense(64, activation='tanh', input_shape=(128,)),
          layers.Dense(9, activation='softmax')
        ])
      elif(type == "LSTM"):
        self.encoder = models.Sequential([
          layers.Dense(32, activation='tanh', input_shape=(18,5)),
          layers.LSTM(64,activation='tanh', return_sequences=True),
          layers.LSTM(64,activation='tanh'),
          layers.Dense(32, activation='tanh', input_shape=(64,)),
          layers.Dense(9, activation='softmax')
        ])

    self.encoder.summary()
    self.autoencoder = models.Sequential([
      self.encoder
    ])
    print("just before compile")
    self.autoencoder.compile(optimizer='adam', loss='categorical_crossentropy',
                             metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.F1Score(average=None, threshold=None)])

    self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    self.x_train = self.x_train.astype('float32')
    self.y_train = self.y_train.astype('float32')
    self.x_val = self.x_val.astype('float32')
    self.y_val = self.y_val.astype('float32')

    self.autoencoder.fit(self.x_train, self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=False)

  def evaluate(self):
    y_pred = self.autoencoder.predict(self.x_val, batch_size=64)
    y_labels = np.argmax(self.y_val, axis=1)
    confusion_matrix = tf.math.confusion_matrix(labels=y_labels, predictions=y_pred.argmax(axis=1)).numpy()
    cateogrial_accuracy = []
    for i in range(9):
      cateogrial_accuracy.append(confusion_matrix[i][i] / sum(confusion_matrix[i]))
    results = self.autoencoder.evaluate(self.x_val, self.y_val, batch_size=64)
    results.append(cateogrial_accuracy)
    return results


load_dotenv()

client = pymongo.MongoClient(os.getenv("MONGO_URI"))
db = client['RNN']
collection = [
    db[('SIMPLE_RNN_1_layer_fullmetric_dense')],
    db[('SIMPLE_RNN_2_layer_fullmetric_dense')],
    db[('GRU_RNN_1_layer_fullmetric_dense')],
    db[('GRU_RNN_2_layer_fullmetric_dense')],
    db[('LSTM_RNN_1_layer_fullmetric_dense')],
    db[('LSTM_RNN_2_layer_fullmetric_dense')]
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

data = pandas.read_csv('./data/EEdata.csv')

words = data['word']
y = data['word']
x = data.iloc[:,2:]
data.iloc[:,2:]
y = np.concatenate([[mapping[i] for i in y]])
x = x.to_numpy()
x = x[:,:90].reshape(900,5,18)
x = x.transpose(0, 2, 1)
x = np.nan_to_num(x)

print(x.shape)

batch_size = 64
epochs = 100
trials = 1

for i in range(trials):
    m1 = researchModel("SimpleRNN", 1, batch_size, epochs, x, y, i)
    results1 = m1.evaluate()
    results1.append(average(results1[2]))
    print(results1)
    # collection[0].insert_one({'model': 'SIMPLE_RNN_1_layer', 'loss': results1[0], 'accuracy': results1[1], 'Macro F1-Score': results1[2]})

# for i in range(trials):
#     m2 = researchModel("SimpleRNN", 2, batch_size, epochs, x, y)
#     results2 = m2.evaluate()
#     collection[1].insert_one({'model': 'SIMPLE_RNN_2_layer', 'loss': results2[0], 'accuracy': results2[1], 'Macro F1-Score': results2[2]})

# for i in range(trials):
#     m3 = researchModel("GRU", 1, batch_size, epochs, x, y)
#     results3 = m3.evaluate()
#     collection[2].insert_one({'model': 'GRU_RNN_1_layer', 'loss': results3[0], 'accuracy': results3[1], 'Macro F1-Score': results3[2]})

# for i in range(trials):
#     m4 = researchModel("GRU", 2, batch_size, epochs, x, y)
#     # results4 = m4.evaluate()
#     # collection[3].insert_one({'model': 'GRU_RNN_2_layer', 'loss': results4[0], 'accuracy': results4[1], 'Macro F1-Score': results4[2]})

# for i in range(trials):
#     m5 = researchModel("LSTM", 1, batch_size, epochs, x, y)
#     # results5 = m5.evaluate()
    # collection[4].insert_one({'model': 'LSTM_RNN_1_layer', 'loss': results5[0], 'accuracy': results5[1], 'Macro F1-Score': results5[2]})

# for i in range(trials):
#     m6 = researchModel("LSTM", 2, batch_size, epochs, x, y)
#     results6 = m6.evaluate()
#     collection[5].insert_one({'model': 'LSTM_RNN_2_layer', 'loss': results6[0], 'accuracy': results6[1], 'Macro F1-Score': results6[2]})


