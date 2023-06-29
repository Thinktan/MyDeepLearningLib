import numpy as np

a = np.array([1 ,2, 3])
b = np.array([2, 3, 4])
# print(a*b)

c = np.array([a.copy(), b.copy()])
# print(c)
# d = c.T
# print(d)
# c[0, 1] = 5
# print(c)
# print(d)

np.random.seed(5)
dropout_ratio = 0.5
xs = c
print('xs: \n', xs)
flg = np.random.rand(*xs.shape) > dropout_ratio
print('fag:\n', flg)

scale = 1 / (1.0 - dropout_ratio)
print('scale: ', scale)

mask = flg.astype(np.float32) * scale
print('mask:\n', mask)

print('ans:\n', xs * mask)