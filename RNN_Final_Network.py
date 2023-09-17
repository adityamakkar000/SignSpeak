import tensorflow as tf
from tensorflow.keras import layers, models
import pandas
import numpy as np

data = pandas.read_csv('./cluster0.RNN_database.csv')

words = data['word']

y = data['word']
x = data.iloc[:,3:]

data.iloc[:,3:]

y = np.eye(15)
y = np.concatenate((y,y))

print(y)

x = x.to_numpy()

print(x.shape)

x = x[:,:300].reshape(30, 10, 30)
x = x.transpose(0, 2, 1)

x = np.nan_to_num(x)
print(x)

# Encoder
encoder = models.Sequential([
    layers.SimpleRNN(64, return_sequences=True, input_shape=(None, 10)),
    layers.SimpleRNN(64)
])
encoder.summary()

# Decoder
decoder = models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(64,)),
    layers.Dense(15, activation='softmax')
])
decoder.summary()

# Combine encoder and decoder into one sequential model
autoencoder = models.Sequential([
    encoder,
    decoder
])

# Display the architecture of the autoencoder
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='mse')  # Using mean squared error loss

# Train the autoencoder
batch_size = 5
epochs = 5

autoencoder.fit(x, y,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x, y))

print(x[0].shape)

print(x[:1])

loss = autoencoder.evaluate(x, y)
print(loss)


autoencoder.save('RNN_model')

# prediction_array = autoencoder.predict(x[:1])[0]


# min_val = 0
# index = -1
# for i in range(0,len(prediction_array)):
#     if prediction_array[i] > min_val:
#         min_val = prediction_array[i]
#         index = i

# print(prediction_array, "    ", index)


