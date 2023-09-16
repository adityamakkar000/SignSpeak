import numpy as np
import pandas as pd


def get_letter_dict():
    cursor = collection.find()
    data = list(cursor)
    letters = [entry["word"] for entry in data]
    resistance_values = [entry["hand"] for entry in data]
    letter_dict = dict(zip(letters, resistance_values))
    return letter_dict

letter_dictionary = get_letter_dict()

test_lst = []
max_size = 20

for letter in ['a', 'b', 'c', 'd', 'e']:
    test_lst.append(lst[letter] + [[1023.0, 1023.0, 1023.0, 1023.0, 1023.0, 1023.0, 1023.0, 1023.0, 1023.0, 1023.0,] for i in range(max_size - len(lst[letter]))])
   
test_lst = np.array(test_lst)
print(test_lst)
