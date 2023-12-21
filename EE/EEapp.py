import serial
import pymongo
from Text import TextToSpeech
import tensorflow as tf
import numpy as np
import pandas as pd

loaded_model = tf.keras.models.load_model('./keras_models/LSTM-DENSE.keras')
print(loaded_model.summary())
tts = TextToSpeech()

words = {
  0: "a",
  1: "b",
  2: "c",
  3: "d",
  4: "e",
  5: "f",
  6: "g",
  7: "h",
  8: "i"
}

def difference_between_arrays(arr1, arr2):
  difference = 0
  for i in range(0, len(arr1)):
    difference += abs(arr1[i] - arr2[i])
  return difference

averageRun = 2
ser = serial.Serial('COM5', 9600)


state = False
stop_amount = 2

while True:

  while state == False:
    line = ser.readline().decode('utf-8').strip()
    temp = line
    print("off", ' ', temp)
    arr = list(map(int,temp.split(' ')))
    sum = 0
    for i in arr:
      sum += i
    if sum < 5000:
      state = True

  while state == True:
    count = 0
    final_arr = []
    # while hand is not in off mode
    while count < stop_amount:

      # setup variables for reading
      sum = 0
      average_reading = [0,0,0,0,0]
      res_arr = [0,0,0,0,0]

      # read data from serial port
      for i in range(0,averageRun):
        line = ser.readline().decode('utf-8').strip()
        if line:
          current_line = line
          current_arr = list(map(int,current_line.split(' ')))
          for i in range(0,5):
            average_reading[i] += current_arr[i]

      # calculate average of readings to minimize noise
      for i in range(0,5):
        res_arr[i] = average_reading[i]/averageRun
        sum += res_arr[i]
      print(res_arr)
      # check if hand is in off mode
      if sum >= 5000:
        count += 1
        print(count)
      else:
        final_arr.append(res_arr)

    unitary = [0,0,0,0,0]
    for i in range(0,18-len(final_arr)):
      final_arr.append(unitary)


    test_array = np.array([final_arr])
    prediction_array = loaded_model(test_array)[0]
    pre = []
    for i in prediction_array:
      pre.append(i)
    print(prediction_array)
    indexs = {}
    for i in range(0, len(pre)):
      indexs[i] = pre[i]
    pre.sort(reverse=True)
    names_list = []
    target = 0
    check = False
    while check == False:
      for i in range(0,9):
        if indexs[i] == pre[target]:
          names_list.append(i)
          target += 1
          break
      if target == 3:
        check =True

    print(words[names_list[0]], " ", words[names_list[1]], " ", words[names_list[2]])

    state = False
