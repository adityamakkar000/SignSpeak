import serial
import pymongo
import os
import math
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from Text import TextToSpeech  # Make sure to import the TextToSpeech class from your module

# Your existing code for database connection and data retrieval
client = pymongo.MongoClient("mongodb+srv://blueishfiend692:EBqcMyVksJPcK2QA@cluster0.so0ju7f.mongodb.net/")
db = client['cluster0']
collection = db[('letters_htn')]

averageRun = 2
ser = serial.Serial('COM5', 9600)

def get_letter_dict():
  cursor = collection.find()
  data = list(cursor)
  letters = [entry["word"] for entry in data]
  resistance_values = [entry["hand"] for entry in data]
  letter_dict = dict(zip(letters, resistance_values))
  return letter_dict

letter_dictionary = get_letter_dict()


# Load the trained PyTorch model and scaler
model = FlexSensorModel()  # Instantiate your trained model here
scaler = MinMaxScaler()  # Instantiate your scaler here
model.load_state_dict(torch.load('cluster0.cluster0.RNN_database.csv'))  # Load the model weights
model.eval()  # Set the model to evaluation mode

# Function to preprocess flex sensor data and predict English word
def predict_word(model, scaler, flex_data):
    flex_data = scaler.transform(flex_data.reshape(1, -1))
    flex_tensor = torch.Tensor(flex_data)
    output = model(flex_tensor)
    predicted_label = torch.argmax(output, dim=1).item()
    predicted_word = index_to_dictionary[predicted_label]
    return predicted_word

final_arr = []

while True:
    try:
        # Your existing code for reading flex sensor data from the serial port
        # ...

        # Assuming res_arr contains your flex sensor data, you can now predict the word
        predicted_word = predict_word(model, scaler, res_arr)
        print(f"Predicted Word: {predicted_word}")

        # Your existing code for speech synthesis
        text = TextToSpeech()
        text.convert_and_play(predicted_word)

        # Your existing code for distance calculation
        # ...

    except KeyboardInterrupt:
        break
