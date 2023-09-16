import serial
import pymongo
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


state = False
stop_amount = 20

while True:

  while state == False:

    line = ser.readline().decode('utf-8').strip()
    a = line
    arr = list(map(int,a.split(' ')))
    sum = 0
    for i in arr:
      sum += i
    if sum < 9700:
      state = True


  while state == True:
    count = 0
    final_arr = []
    while count < stop_amount:
      line = ser.readline().decode('utf-8').strip()
      a = line
      arr = list(map(int,a.split(' ')))
      sum = 0
      for i in arr:
        sum += i
      if sum >= 9700:
        count += 1
      else:
        final_arr.append(arr)

    # use prediction then
    #
    print("stop")
    state = False









