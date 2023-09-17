import tensorflow as tf
from tensorflow.keras import layers, models
import pandas
import numpy as np
from sklearn.model_selection import train_test_split

mapping = {'hello': np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
           'a': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
           's': np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
           'l': np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
           'speech': np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
           'this': np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
           'recognition': np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
           'yes': np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
           'no': np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
           'wrong': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])}


data = pandas.read_csv('./cluster0.new_RNN_database.csv')
words = data['word']

y = data['word']
x = data.iloc[:,3:]

data.iloc[:,3:]


y = np.concatenate([[mapping[i] for i in y]])

print(y.shape)

x = x.to_numpy()


x = x[:,:210].reshape(127, 10, 21)
x = x.transpose(0, 2, 1)

print(x.shape)
x = np.nan_to_num(x)
print(x.shape)

# Encoder
encoder = models.Sequential([
    layers.SimpleRNN(64, return_sequences=True, input_shape=(None, 10)),
    layers.SimpleRNN(64)
])
encoder.summary()

# Decoder
decoder = models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(64,)),
    layers.Dense(10, activation='softmax')
])
decoder.summary()

# Combine encoder and decoder into one sequential model
autoencoder = models.Sequential([
    encoder,
    decoder
])

# Display the architecture of the autoencoder
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])  # Using mean squared error loss

# Train the autoencoder
batch_size = 3
epochs = 20

# spilt 80 20 into train and vlaidation
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
autoencoder.fit(x_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_val, y_val))

# Evaluate the autoencoder on the validation data
autoencoder.evaluate(x_val, y_val)



autoencoder.save('RNN_model.keras')


prediction_array = autoencoder.predict(x[:1])[0]

print (x[:1])


min_val = 0
index = -1
for i in range(0,len(prediction_array)):
    if prediction_array[i] > min_val:
        min_val = prediction_array[i]
        index = i

print(prediction_array, "    ", index)


