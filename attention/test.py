import numpy as np

t = np.arange(24).reshape([2, 3, 4])
print(t)
print(t.shape, '\n')
print(t[:,-1])
print(t[:,-1].shape)
