import serial
import pymongo
import tensorflow as tf
import numpy as np
import pandas as pd
import serial
import pymongo
import os


client = pymongo.MongoClient('mongodb+srv://aditymakkar000:LvEfNChu8b9k7nyL@eedata.l1ghxko.mongodb.net/')
db = client['cluster0']
collection = db[('EEDatabase')]



print("connected to database")

averageRun = 2
ser = serial.Serial('COM5', 9600)

state = False
stop_amount = 2
word = 'i'
word_count = 0

def insert_data(data):
  collection.insert_one(data)

while word_count < 100:

  while state == False:

    line = ser.readline().decode('utf-8').strip()
    temp = line
    print("off", ' ', temp)
    arr = list(map(int,temp.split(' ')))
    sum = 0
    # print(arr)
    for i in arr:
      sum += i
    if sum < 5000:
      state = True

  while  state == True:
    count = 0
    final_arr = []
    while count < stop_amount:

      sum = 0
      average_reading = [0,0,0,0,0]
      res_arr = [0,0,0,0,0]

      for i in range(0,averageRun):
        line = ser.readline().decode('utf-8').strip()
        if line:
          current_line = line
          current_arr = list(map(int,current_line.split(' ')))
          for i in range(0,5):
            average_reading[i] += current_arr[i]

      for i in range(0,5):
        res_arr[i] = average_reading[i]/averageRun
        sum += res_arr[i]

      print("on", ' ', res_arr)
      # print(res_arr)
      # print(sum)

      if sum >= 5000:
        count += 1
        print(count)
      else:
        final_arr.append(res_arr)

    data = {"word": word, "hand": final_arr}
    insert_data(data)
    word_count += 1
    print(word_count)
    state = False



