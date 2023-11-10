import serial
import pymongo
from Text import TextToSpeech
import tensorflow as tf
import numpy as np
import pandas as pd

<<<<<<< HEAD:app.py
loaded_model = tf.keras.models.load_model('RNN_model_new_4.keras')
print(loaded_model.summary())
tts = TextToSpeech()

client = pymongo.MongoClient("mongodb+srv://blueishfiend692:EBqcMyVksJPcK2QA@cluster0.so0ju7f.mongodb.net/")
db = client['cluster0']
=======
loaded_model = tf.keras.models.load_model('RNN_model.keras')
print(loaded_model.summary())
tts = TextToSpeech()

client = pymongo.MongoClient("database/")
>>>>>>> e278c16ed44bd4c787730812fa4135426b066e28:final.py
collection = db[('new_RNN_database')]

words = []

for entry in collection.find():
  if entry["word"] not in words:
    words.append(entry["word"])


def difference_between_arrays(arr1, arr2):
  difference = 0
  for i in range(0, len(arr1)):
    difference += abs(arr1[i] - arr2[i])
  return difference

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


state = False
stop_amount = 8
<<<<<<< HEAD:app.py


=======
>>>>>>> e278c16ed44bd4c787730812fa4135426b066e28:final.py

while True:

  while state == False:
    print("off ")
    line = ser.readline().decode('utf-8').strip()
    a = line
    arr = list(map(int,a.split(' ')))
    sum = 0
    for i in arr:
      sum += i
    if sum < 9500:
      state = True


  while state == True:

    count = 0
    final_arr = []
    while count < stop_amount:
      print("on")
      average_reading = [0,0,0,0,0,0,0,0,0,0]
      res_arr = [0,0,0,0,0,0,0,0,0,0]

      for i in range(0,averageRun):
          line = ser.readline().decode('utf-8').strip()
          if line:
            a = line
            arr = list(map(int,a.split(' ')))
            # print(arr)
            for i in range(0,10):
              average_reading[i] += arr[i]
      for i in range(0,10):
        res_arr[i] = average_reading[i]/averageRun
      sum = 0
      print(res_arr)

      for i in res_arr:
        sum += i
      print(sum)
      if sum >= 9500:
        count += 1
      else:
        final_arr.append(res_arr)
      print(count)

      unitary = [0,0,0,0,0,0,0,0,0,0]
      diff = 21 - len(final_arr)
      for i in range(0, diff):
        final_arr.append(unitary)

    test_array = np.array([final_arr])
    prediction_array = loaded_model(test_array)[0]
    print(prediction_array)
    min_val = 0
    index = -1
    for i in range(0,len(prediction_array)):
      if prediction_array[i] > min_val:
        min_val = prediction_array[i]
        index = i

    word = words[index]
    tts.convert_and_play(word)
    print("stop")
    state = False









