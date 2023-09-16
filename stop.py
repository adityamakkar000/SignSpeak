import serial
import pymongo
import os
import math
from Text import TextToSpeech

client = pymongo.MongoClient("mongodb+srv://blueishfiend692:EBqcMyVksJPcK2QA@cluster0.so0ju7f.mongodb.net/")
db = client['cluster0']
collection = db[('letters_average')]


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


state = 0

while True:
  line = ser.readline().decode('utf-8').strip()
  count = 0:

  while count < 0:


