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
    for i in range(0,len(res_arr)):
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

# create a for loop put every element in letter dictionary and an empty array
for i in letter_dictionary:
  letter_distance[i] = [0,0]

print(letter_distance)

for j in letter_dictionary:
    letter = letter_dictionary[j]
    for i in range(0,len(letter)):
      time_arr = letter[i]
      hand_1 = 0
      hand_2 = 0
      # print(time_arr)
      for k in range(0,5):
        power = 5*k
        hand_1 += pow(10, power) * time_arr[k]
        hand_2 += pow(10, power) * time_arr[k+5]
      letter_distance[j][0] += hand_1
      letter_distance[j][1] += hand_2

print(letter_distance)


sequence_distance = [0,0]

for i in range(0, len(final_arr)):
  instant_array = final_arr[i]
  for j in range(0,5):
      power = 5*j
      hand_1 += pow(10, power) * instant_array[j]
      hand_2 += pow(10, power) * instant_array[j+5]
  sequence_distance[0] += hand_1
  sequence_distance[1] += hand_2
  # print()
    # put in dictionary Let

p = 2

hand_one = sequence_distance[0]
hand_two = sequence_distance[1]

for i in letter_distance:
  index = list(letter_distance).index(i)
  distances_for_letter = letter_distance[i]
  distance_arr[index] = pow(pow(abs(distances_for_letter[0]-hand_1),2) + pow(abs(distances_for_letter[1]-hand_2),2) ,1/2)

min_dis = 10000000000000000000000000000000000000000000000000000000000000000
min_index = -1
for i in range(0,len(distance_arr)):
  if(distance_arr[i] < min_dis):
    min_dis = distance_arr[i]
    min_index = i

index_to_dictionary = list(letter_dictionary)[min_index]


print(min_index)
print(distance_arr)



