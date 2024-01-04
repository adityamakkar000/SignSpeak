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


# get average of list
def average(lst):
    return sum(lst) / len(lst)

# upload data
def data_upload(model, collection):
  cms = np.array(model.confusion_matrixs).tolist()
  data  = {'model': model.type,'confusion_matrix': cms}
  collection.insert_one(data)

# get encoder model
class encoderModel:

  def __init__(self, type, stacks, dense):
    self.seed = 42
    np.random.seed(self.seed)
    tf.random.set_seed(self.seed)
    random.seed(self.seed)
    os.environ['PYTHONHASHSEED']=str(self.seed)


    self.RNNNeurons = 128
    self.DenseNeurons = 128

    if(dense == True):
      self.encoder = models.Sequential([
        layers.Dense(self.DenseNeurons, activation='tanh', input_shape=(9,5))
      ])
    else:
      self.encoder = models.Sequential([])

    if(stacks == 2):
      if(type == "SimpleRNN"):
        self.encoder.add(layers.SimpleRNN(self.RNNNeurons,activation='tanh',
                                          return_sequences=True,input_shape=(9,5)))
      if(type == "GRU"):
        self.encoder.add(layers.GRU(self.RNNNeurons,activation='tanh',
                                    return_sequences=True, input_shape=(9,5)))
      if(type == "LSTM"):
        self.encoder.add(layers.LSTM(self.RNNNeurons,activation='tanh',
                                     return_sequences=True, input_shape=(9,5)))

    if(type == "SimpleRNN"):
      self.encoder.add(layers.SimpleRNN(self.RNNNeurons,
                                        activation='tanh', input_shape=(9,5)))
    elif(type == "GRU"):
      self.encoder.add(layers.GRU(self.RNNNeurons,
                                  activation='tanh', input_shape=(9,5)))
    elif(type == "LSTM"):
      self.encoder.add(layers.LSTM(self.RNNNeurons,
                                   activation='tanh', input_shape=(9,5)))


    self.decoder = models.Sequential([
      layers.Dense(self.DenseNeurons,
                   activation='tanh', input_shape=(self.RNNNeurons,)),
      layers.Dense(10, activation='softmax')
    ])

    self.autoencoder = models.Sequential(self.encoder.layers + self.decoder.layers)

# define the research  model
class researchModel:

  def __init__(self, type, stacks, x, y, epochs, batch_size, dense):
    self.seed = 42
    np.random.seed(self.seed)
    tf.random.set_seed(self.seed)
    random.seed(self.seed)
    os.environ['PYTHONHASHSEED']=str(self.seed)

    self.type = type
    self.final_results = []
    self.confusion_matrixs = []
    self.spilts = 5
    kfold = StratifiedKFold(n_splits=self.spilts, shuffle=True, random_state=self.seed)
    autoencoder = encoderModel(type, stacks, dense).autoencoder
    autoencoder.summary()
    with open('param.txt', 'a') as f:
        f.write(str(autoencoder.count_params()) + type  +'\n')
    y_label_encoded = np.argmax(y, axis=1)
    for i, (train, test) in enumerate(kfold.split(x, y_label_encoded)):

      x_train = x[train].astype('float32')
      y_train = y[train].astype('float32')
      x_val = x[test].astype('float32')
      y_val = y[test].astype('float32')

      autoencoder = encoderModel(type, stacks, dense).autoencoder
      autoencoder.summary()

      autoencoder.compile(optimizer='adam', loss='categorical_crossentropy',
                                metrics=[keras.metrics.CategoricalAccuracy(),
                                         keras.metrics.F1Score(average=None, threshold=None)])
      autoencoder.fit(x_train, y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  shuffle=False)

      y_pred = autoencoder.predict(x_val, batch_size=64)
      y_labels = np.argmax(y_val, axis=1)
      confusion_matrix = tf.math.confusion_matrix(labels=y_labels,
                                                  predictions=y_pred.argmax(axis=1)).numpy()
      self.confusion_matrixs.append(confusion_matrix)
      cat_accuracy = []
      for i in range(10):
        TP = confusion_matrix[i][i]
        cat_accuracy.append(TP/sum(confusion_matrix[i]))
      results = autoencoder.evaluate(x_val, y_val, batch_size=64)
      accuracy = results[1]
      with open('test.txt', 'a') as f:
          f.write(str(accuracy) +'\n')
      results.append(average(results[2]))
      results.append(cat_accuracy)
      self.final_results.append(results)


# data  processing
load_dotenv()
client = pymongo.MongoClient(os.getenv("MONGO_URI"))
db = client['RNN_4_additional']
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

mapping = {'a': [1,0,0,0,0,0,0,0,0,0],
           'b': [0,1,0,0,0,0,0,0,0,0],
           'c': [0,0,1,0,0,0,0,0,0,0],
           'd': [0,0,0,1,0,0,0,0,0,0],
           'e': [0,0,0,0,1,0,0,0,0,0],
           'f': [0,0,0,0,0,1,0,0,0,0],
           'g': [0,0,0,0,0,0,1,0,0,0],
           'h': [0,0,0,0,0,0,0,1,0,0],
           'i': [0,0,0,0,0,0,0,0,1,0],
         'none':[0,0,0,0,0,0,0,0,0,1]}

data = pandas.read_csv('./Data/Data.csv')

words = data['word']
y = data['word']
x = data.iloc[:,2:]
data.iloc[:,2:]
y = np.concatenate([[mapping[i] for i in y]])
x = x.to_numpy()
x = x[:,:45].reshape(1000,5,9)
x = x.transpose(0, 2, 1)
x = np.nan_to_num(x)

print(x.shape)

batch_size = 64
epochs = 1000

# training
print("starting training ...")

models = {
  # "m1": researchModel("SimpleRNN", 1, x, y, epochs,batch_size, dense=False),
  "m2": researchModel("SimpleRNN", 2, x, y, epochs,batch_size, dense=False),
  "m3": researchModel("GRU", 1, x, y, epochs,batch_size, dense=False),
  # "m4": researchModel("GRU", 2, x, y, epochs,batch_size, dense=False),
  # "m5": researchModel("LSTM", 1, x, y, epochs,batch_size, dense=False),
  # "m6": researchModel("LSTM", 2, x, y, epochs,batch_size, dense=False),
  # "m7": researchModel("SimpleRNN", 1, x, y, epochs,batch_size, dense=True),
  # "m8": researchModel("SimpleRNN", 2, x, y, epochs,batch_size, dense=True),
  "m9": researchModel("GRU", 1, x, y, epochs,batch_size, dense=True),
  "m10": researchModel("GRU", 2, x, y, epochs,batch_size, dense=True),
  # # "m11": researchModel("LSTM", 1, x, y, epochs,batch_size, dense=True),
  # "m12":researchModel("LSTM", 2, x, y, epochs,batch_size, dense=True)
}

for i in models:
  data_upload(models[i], collection[int(i[1:])-1])
