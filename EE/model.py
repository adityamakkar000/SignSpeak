import tensorflow as tf
from tensorflow.keras import layers, models
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
import pymongo
import keras

class researchModel:

  def __init__(self, type, stacks, batch_size, epochs, x, y):
    if(stacks == 1):
      if(type == "SimpleRNN"):
        self.encoder = models.Sequential([
          layers.Dense(32, activation='tanh', input_shape=(18,5)),
          layers.SimpleRNN(16,activation='tanh'),
          layers.Dense(32, activation='tanh', input_shape=(64,)),
          layers.Dense(9, activation='softmax', input_shape=(64,))
        ])
      elif(type == "GRU"):
        self.encoder = models.Sequential([
          layers.Dense(32, activation='tanh', input_shape=(18,5)),
          layers.GRU(16,activation='tanh'),
          layers.Dense(32, activation='tanh', input_shape=(64,)),
          layers.Dense(9, activation='softmax', input_shape=(64,))
        ])
      elif(type == "LSTM"):
        self.encoder = models.Sequential([
          layers.Dense(32, activation='tanh', input_shape=(18,5)),
          layers.LSTM(16,activation='tanh'),
          layers.Dense(32, activation='tanh', input_shape=(64,)),
          layers.Dense(9, activation='softmax', input_shape=(64,))
        ])
    elif(stacks == 2):
      if(type == "SimpleRNN"):
        self.encoder = models.Sequential([
          layers.Dense(32, activation='tanh', input_shape=(18,5)),
          layers.SimpleRNN(16,activation='tanh', return_sequences=True),
          layers.SimpleRNN(16,activation='tanh'),
          layers.Dense(32, activation='tanh', input_shape=(64,)),
          layers.Dense(9, activation='softmax', input_shape=(64,))
        ])
      elif(type == "GRU"):
        self.encoder = models.Sequential([
          layers.Dense(32, activation='tanh', input_shape=(18,5)),
          layers.GRU(16,activation='tanh', return_sequences=True),
          layers.GRU(16,activation='tanh'),
          layers.Dense(32, activation='tanh', input_shape=(64,)),
          layers.Dense(9, activation='softmax', input_shape=(64,))
        ])
      elif(type == "LSTM"):
        self.encoder = models.Sequential([
          layers.Dense(32, activation='tanh', input_shape=(18,5)),
          layers.LSTM(16,activation='tanh', return_sequences=True),
          layers.LSTM(16,activation='tanh'),
          layers.Dense(32, activation='tanh', input_shape=(64,)),
          layers.Dense(9, activation='softmax', input_shape=(64,))
        ])


    self.autoencoder = models.Sequential([
      self.encoder
    ])
    print("just before compile")
    self.autoencoder.compile(optimizer='adam', loss='categorical_crossentropy',
                             metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.F1Score(average='macro', threshold=None)])

    self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    self.x_train = self.x_train.astype('float32')
    self.y_train = self.y_train.astype('float32')
    self.x_val = self.x_val.astype('float32')
    self.y_val = self.y_val.astype('float32')

    self.autoencoder.fit(self.x_train, self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True)

  def evaluate(self):
    return self.autoencoder.evaluate(self.x_val, self.y_val, batch_size=64)

