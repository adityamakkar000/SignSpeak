import numpy as np

arr1 = np.arange(5)[:, np.newaxis]
arr2 = np.arange(5,10)[np.newaxis,:]
print(arr2)
