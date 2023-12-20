import tensorflow as tf
from tensorflow.keras import layers, models
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
import pymongo


class researchModel:

  def __init__(self, type, stacks, batch_size, epochs, x, y):
    if(stacks == 1):
      if(type == "SimpleRNN"):
        self.encoder = models.Sequential([
          layers.SimpleRNN(64,activation='tanh'),
          layers.Dense(64, activation='tanh', input_shape=(64,)),
          layers.Dense(9, activation='softmax')
        ])
      elif(type == "GRU"):
        self.encoder = models.Sequential([
          layers.GRU(64,activation='tanh'),
          layers.Dense(64, activation='tanh', input_shape=(64,)),
          layers.Dense(9, activation='softmax')
        ])
      elif(type == "LSTM"):
        self.encoder = models.Sequential([
          layers.LSTM(64,activation='tanh'),
          layers.Dense(64, activation='tanh', input_shape=(64,)),
          layers.Dense(9, activation='softmax')
        ])
    elif(stacks == 2):
      if(type == "SimpleRNN"):
        self.encoder = models.Sequential([
          layers.SimpleRNN(64,activation='tanh', return_sequences=True),
          layers.SimpleRNN(64,activation='tanh'),
          layers.Dense(64, activation='tanh', input_shape=(64,)),
          layers.Dense(9, activation='softmax')
        ])
      elif(type == "GRU"):
        self.encoder = models.Sequential([
          layers.GRU(64,activation='tanh', return_sequences=True),
          layers.GRU(64,activation='tanh'),
          layers.Dense(64, activation='tanh', input_shape=(64,)),
          layers.Dense(9, activation='softmax')
        ])
      elif(type == "LSTM"):
        self.encoder = models.Sequential([
          layers.LSTM(64,activation='tanh', return_sequences=True),
          layers.LSTM(64,activation='tanh'),
          layers.Dense(64, activation='tanh', input_shape=(64,)),
          layers.Dense(9, activation='softmax')
        ])


    self.autoencoder = models.Sequential([
      self.encoder
    ])

    self.autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    self.autoencoder.fit(x_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True)

  def evaluate(self, x, y):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    return self.autoencoder.evaluate(x_val, y_val)

