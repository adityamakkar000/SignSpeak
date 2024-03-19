import 
import pandas
import numpy as np


data = pandas.read_csv('./Data/Data.csv')

mapping = {'a': [1,0,0,0,0,0,0,0,0,0],
           'b': [0,1,0,0,0,0,0,0,0,0],
           'c': [0,0,1,0,0,0,0,0,0,0],
           'd': [0,0,0,1,0,0,0,0,0,0],
           'e': [0,0,0,0,1,0,0,0,0,0],
           'f': [0,0,0,0,0,1,0,0,0,0],
           'g': [0,0,0,0,0,0,1,0,0,0],
           'h': [0,0,0,0,0,0,0,1,0,0],
           'i': [0,0,0,0,0,0,0,0,1,0],
         'none':[0,0,0,0,0,0,0,0,0,1]}

words = data['word']
y = data['word']
x = data.iloc[:,2:]
data.iloc[:,2:]
y = np.concatenate([[mapping[i] for i in y]])
x = x.to_numpy()
x = x[:,:45].reshape(1000,5,9)
x = x.transpose(0, 2, 1)
x = np.nan_to_num(x)

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

print(x, '\n', y)
