import tensorflow as tf
from tensorflow.keras import layers, models
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pymongo
import os
from dotenv import load_dotenv
from sklearn.metrics import ConfusionMatrixDisplay
import keras
import matplotlib.pyplot as plt
import random


# get average of list
def average(lst):
    return sum(lst) / len(lst)

# get encoder model
class encoderModel:
  def __init__(self, type, stacks):
    self.seed = 42
    np.random.seed(self.seed)
    tf.random.set_seed(self.seed)
    random.seed(self.seed)
    os.environ['PYTHONHASHSEED']=str(self.seed)

    if(stacks == 1):
        if(type == "SimpleRNN"):
          self.encoder = models.Sequential([
            layers.Dense(64, activation='tanh', input_shape=(18,5)),
            layers.SimpleRNN(64,activation='tanh', input_shape=(18,5)),
            layers.Dense(64, activation='tanh', input_shape=(64,)),
            layers.Dense(9, activation='softmax')
          ])
        elif(type == "GRU"):
          self.encoder = models.Sequential([
            layers.Dense(64, activation='tanh', input_shape=(18,5)),
            layers.GRU(64,activation='tanh'),
            layers.Dense(64, activation='tanh', input_shape=(64,)),
            layers.Dense(9, activation='softmax')
          ])
        elif(type == "LSTM"):
          self.encoder = models.Sequential([
            # layers.Dense(64, activation='tanh', input_shape=(18,5)),
            layers.LSTM(64,activation='tanh', input_shape=(18,5)),
            layers.Dense(64, activation='tanh', input_shape=(64,)),
            layers.Dense(9, activation='softmax')
          ])
    elif(stacks == 2):
        if(type == "SimpleRNN"):
          self.encoder = models.Sequential([
            layers.Dense(64, activation='tanh', input_shape=(18,5)),
            layers.SimpleRNN(64,activation='tanh', return_sequences=True, input_shape=(18,5)),
            layers.SimpleRNN(64,activation='tanh'),
            layers.Dense(64, activation='tanh', input_shape=(64,)),
            layers.Dense(9, activation='softmax')
          ])
        elif(type == "GRU"):
          self.encoder = models.Sequential([
            # layers.Dense(64, activation='tanh', input_shape=(18,5)),
            layers.GRU(64,activation='tanh', return_sequences=True, input_shape=(18,5)),
            layers.GRU(64,activation='tanh'),
            layers.Dense(64, activation='tanh', input_shape=(64,)),
            layers.Dense(9, activation='softmax')
          ])
        elif(type == "LSTM"):
          self.encoder = models.Sequential([
            # layers.Dense(64, activation='tanh', input_shape=(18,5)),
            layers.LSTM(64,activation='tanh', return_sequences=True, input_shape=(18,5)),
            layers.LSTM(64,activation='tanh'),
            layers.Dense(64, activation='tanh', input_shape=(64,)),
            layers.Dense(9, activation='softmax')
          ])


# define the keras model

class researchModel:

  def __init__(self, type, stacks, x, y, epochs, batch_size):
    self.seed = 42
    np.random.seed(self.seed)
    tf.random.set_seed(self.seed)
    random.seed(self.seed)
    os.environ['PYTHONHASHSEED']=str(self.seed)

    self.final_results = []
    self.confusion_matrixs = []
    self.spilts = 5 # change to 10 when needed
    kfold = KFold(n_splits=self.spilts, shuffle=True)

    for i, (train, test) in enumerate(kfold.split(x, y)):

      print(train)
      print("\n\n")
      print(test)
      self.encoder = encoderModel(type, stacks).encoder
      self.encoder.summary()

      x_train = x[train].astype('float32')
      y_train = y[train].astype('float32')
      x_val = x[test].astype('float32')
      y_val = y[test].astype('float32')
      autoencoder = models.Sequential([
        self.encoder
      ])

      autoencoder.compile(optimizer='adam', loss='categorical_crossentropy',
                                metrics=[keras.metrics.CategoricalAccuracy(),
                                         keras.metrics.F1Score(average=None, threshold=None)])
      autoencoder.fit(x_train, y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  shuffle=False)

      y_pred = autoencoder.predict(x_val, batch_size=64)
      y_labels = np.argmax(y_val, axis=1)
      confusion_matrix = tf.math.confusion_matrix(labels=y_labels, predictions=y_pred.argmax(axis=1)).numpy()
      self.confusion_matrixs.append(confusion_matrix)
      cateogrial_accuracy = []
      for i in range(9):
        cateogrial_accuracy.append(confusion_matrix[i][i] / sum(confusion_matrix[i]))
      results = autoencoder.evaluate(x_val, y_val, batch_size=64)
      results.append(average(results[2]))
      results.append(cateogrial_accuracy)
      self.final_results.append(results)


# data  processing

load_dotenv()
client = pymongo.MongoClient(os.getenv("MONGO_URI"))
db = client['RNN_2']
collection = [
    db[('SIMPLE_1_layer_dense')],
    db[('SIMPLE_2_layer_dense')],
    db[('GRU_1_layer_dense')],
    db[('GRU_2_layer_dense')],
    db[('LSTM_1_layer_dense')],
    db[('LSTM_2_layer_dense')]
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
epochs = 1000

# training
print("starting training ...")

m1 = researchModel("SimpleRNN", 2, x, y, epochs,batch_size)
confusion_matrixs = m1.confusion_matrixs

cm = np.zeros((9,9))
for i in confusion_matrixs:
  cm += i
cm /= 5
# total = 0
# for i in range(9):
#   cm[i] /= sum(cm[i])


disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['a','b','c','d','e','f','g','h','i'])

disp.plot()
plt.show()

