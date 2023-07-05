import numpy as np

# t = np.arange(24).reshape([2, 3, 4])
t = np.arange(8).reshape([2, 1, 4])
print(t)
# print(t.shape, '\n')
# print(t[:,-1])
# print(t[:,-1].shape)
print('----')
# print(t[:, :-1])
d1 = t.reshape(2, 4)
print(d1)