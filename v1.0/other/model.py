import tensorflow as tf
from tensorflow.keras import layers, models
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
import pymongo
import keras

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
x = x[:,:45].reshape(900,5,9)
x = x.transpose(0, 2, 1)
x = np.nan_to_num(x)

print(x.shape)

batch_size = 64
epochs = 1000


encoder = models.Sequential([
          # layers.Dense(64, activation='tanh', input_shape=(18,5)),
          layers.SimpleRNN(128,activation='tanh', return_sequences=True, input_shape=(None,5)),
          layers.SimpleRNN(128,activation='tanh'),
          layers.Dense(64, activation='tanh', input_shape=(128,)),
          layers.Dense(9, activation='softmax')
 ])

encoder.summary()

autoencoder = models.Sequential([
      encoder
])

autoencoder.compile(optimizer='adam', loss='categorical_crossentropy',
  metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.F1Score(average='macro', threshold=None)])

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_val = x_val.astype('float32')
y_val = y_val.astype('float32')

autoencoder.fit(x_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True)

autoencoder.evaluate(x_val, y_val, batch_size=batch_size)

autoencoder.save('./keras_models/LSTM-DENSE.keras')

