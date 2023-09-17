import serial
import pymongo
import os
import math
from Text import TextToSpeech

client = pymongo.MongoClient("database/")
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


final_arr = []
while True:
  try:
    res_arr = [0,0,0,0,0,0,0,0,0,0]
    for i in range(0, averageRun):
      line = ser.readline().decode('utf-8').strip()  # Read a line of text from the serial port
      if line:  # Check if the line is not empty
        a = line
              # print(a)
              #covert string to integer array
        arr = list(map(int, a.split(' ')))
        for i in range(0,len(res_arr)):
          res_arr[i] += arr[i]
    for i in range(0,5):
      res_arr[i] = res_arr[i]/averageRun
    print(res_arr)
    final_arr.append(res_arr)
  except KeyboardInterrupt:
    break

distance_arr = []

for i in range(0,len(letter_dictionary)):
  distance_arr.append(0)

power = 2

for i in range(0, len(final_arr)):
  for j in letter_dictionary:
    letter = letter_dictionary[j]
    time_arr = letter[i]
    distance_arr[list(letter_dictionary).index(j)] += pow(
    (pow(abs(final_arr[i][0] - time_arr[0]),power)
    + pow(abs(final_arr[i][1] - time_arr[1]),power)
    + pow(abs(final_arr[i][2] - time_arr[2]),power)
    + pow(abs(final_arr[i][3] - time_arr[3]),power)
    + pow(abs(final_arr[i][4] - time_arr[4]),power)), (1/power)
    )


min_dis = 1000000000000
min_index = -1

for i in range(0,len(distance_arr)):
  if(distance_arr[i] < min_dis):
    min_dis = distance_arr[i]
    min_index = i

index_to_dictionary = list(letter_dictionary)[min_index]
print(index_to_dictionary)

text = TextToSpeech()
text.convert_and_play(index_to_dictionary)

print(min_index)
print(distance_arr)
