import numpy as np

dW = np.array([[1, 2, 3],
               [1, 3, 4],
               [2, 4, 6],
               [1, 5, 7],
               [6, 7, 8]])

dW2 = np.array([[1, 2, 3],
               [1, 3, 4],
               [2, 4, 6],
               [1, 5, 7],
               [6, 7, 8]])

idx = np.array([0, 2, 0, 4])

dout = np.array([[1, 1, 1],
                 [3, 3, 3],
                 [6, 6, 6],
                 [9, 9, 9]])


for i, word_id in enumerate(idx):
    dW[word_id] += dout[i]

print(dW, '\n')

np.add.at(dW2, idx, dout)
print(dW2)

x = np.arange(12).reshape(3, 4)
print(x)
print(x.shape)
# print(np.sum(x, axis=1))
# print(np.sum(x, axis=1).shape)

y = np.array([1, 2, 3])
z = y.reshape(y.shape[0], 1)
print(y)

print(x*z)

print(np.power(y, .75))

