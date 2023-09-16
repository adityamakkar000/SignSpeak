import serial
import pymongo
import os
import math
from Text import TextToSpeech

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
  distance_arr.append([0,0])


# create empyt dictionary
letter_distance = dict()

for j in letter_dictionary:
    letter = letter_dictionary[j]
    time_arr = letter[i]
    hand_1 = 0
    hand_2 = 0
    for i in range(0,5):
      power = 3*i
      hand_1 += pow(10, power) * time_arr[i]
      hand_2 += pow(10, power) * time_arr[i+5]
    letter_distance[j] = [hand_1, hand_2]

print(letter_distance)

for i in range(0, len(final_arr)):
  print(final_arr[i])
  for j in range(0,5):
      power = 3*j
      hand_1 += pow(10, power) * time_arr[j]
      hand_2 += pow(10, power) * time_arr[j+5]
    distance_arr[j] = [hand_1, hand_2]
    # put in dictionary Let



