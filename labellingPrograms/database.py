import serial
import pymongo
import os

# connect and setup mongodb database
client = pymongo.MongoClient("database")
db = client['cluster0']
collection = db[('new_RNN_database')]

# average to smooth out data
averageRun = 2

def insert_data(data):
  collection.insert_one(data)

ser = serial.Serial('COM5', 9600)

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
    for i in range(0,len(res_arr)):
      res_arr[i] = res_arr[i]/averageRun
    print(res_arr)
    final_arr.append(res_arr)
  except KeyboardInterrupt:
    word = input("Enter a word: ")
    removed_arr = []
    for i in final_arr:
      sum = 0
      for j in i:
        sum += j
      if sum >= 9700:
        removed_arr.append(i)
    for i in removed_arr:
      final_arr.remove(i)
    data = {"word": word, "hand": final_arr}
    print(data)
    insert_data(data)
    key = 'b'
    break


