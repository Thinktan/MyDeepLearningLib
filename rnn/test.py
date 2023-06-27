
import numpy as np

x = np.array([[1, 2, 3], [2, 3, 4]])
print(np.sum(x, axis=0).shape)
print(x[:,1].shape)

print(np.argmax(x, axis=1))
print(np.argmax(np.array([0, 0, 1])))
print(x != -1)
