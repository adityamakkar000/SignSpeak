import serial
import pymongo
import os

# connect and setup mongodb database
client = pymongo.MongoClient("mongodb+srv://blueishfiend692:EBqcMyVksJPcK2QA@cluster0.so0ju7f.mongodb.net/")
db = client['cluster0']
collection = db[('letters_htn')]

# average to smooth out data
averageRun = 10

def insert_data(data):
  if collection.count_documents({"word": data["word"]}) == 0: # if word does not exist in database
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
    for i in range(0,5):
      res_arr[i] = res_arr[i]/averageRun
    print(res_arr)
    final_arr.append(res_arr)
  except KeyboardInterrupt:
    word = input("Enter a word: ")
    data = {"word": word, "hand": final_arr}
    insert_data(data)
    key = 'b'
    break


