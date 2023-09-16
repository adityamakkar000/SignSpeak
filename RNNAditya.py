import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pymongo

client = pymongo.MongoClient("mongodb+srv://blueishfiend692:EBqcMyVksJPcK2QA@cluster0.so0ju7f.mongodb.net/")
db = client['cluster0']
collection = db[('RNN_database')]

def get_letter_dict():
    cursor = collection.find()
    data = list(cursor)
    letters = [entry["word"] for entry in data]
    resistance_values = [entry["hand"] for entry in data]
    letter_dict = dict(zip(letters, resistance_values))
    return letter_dict

dictionary = get_letter_dict()



sequences = []
words = []
for i in dictionary:
    words.append(i)
    sequences.append(dictionary[i])

