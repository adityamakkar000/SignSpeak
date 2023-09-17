import serial
import pymongo
import os

# connect and setup mongodb database
client = pymongo.MongoClient("database")
db = client['cluster0']
collection = db[('letters_average')]

# average to smooth out data


def insert_data(data):
  if collection.count_documents({"word": data["word"]}) == 0: # if word does not exist in database
    collection.insert_one(data)

ser = serial.Serial('COM5', 9600)

final_arr = []
res_arr = [0,0,0,0,0,0,0,0,0,0]
count = 0
while True:
  try:
    line = ser.readline().decode('utf-8').strip()

    if line:
      count += 1
      a = line
      # print(line)
      arr = list(map(int, a.split(' ')))
      for i in range(0,len(res_arr)):
        res_arr[i] += arr[i]

  except KeyboardInterrupt:
    for i in range(0,len(res_arr)):
      final_arr.append(res_arr[i]/count)
    word = input("Enter a word: ")
    data = {"word": word, "hand": final_arr}
    insert_data(data)
    break
