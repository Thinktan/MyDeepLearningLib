# coding: utf-8

import numpy as np

x = np.array([1, 2, -3])
y = x
y[1] = 4
print(x)

print(np.maximum(0, x))

print('--------softmax1---------')

# x = np.arange(0, 6).reshape(2, 3)
x = np.array([[1, 5, 3], [2, 4, 6]])
print(x, '\n')
# print(x.max(axis=1), '\n')
print(x.max(axis=1, keepdims=True), '\n')
# print(x.max(axis=0), '\n')
# print(x.max(axis=0, keepdims=True), '\n')

# print(x - x.max(axis=1, keepdims=True), '\n')

x = x - x.max(axis=1, keepdims=True)
print(x, '\n')

x = np.exp(x)
print(x, '\n')

print(x.sum(axis=1, keepdims=True), '\n')

x /= x.sum(axis=1, keepdims=True)
print(x, '\n')

print('--------softmax2---------')
x = np.array([1, 5, 3])
print(np.max(x), '\n')
print(x - np.max(x), '\n')
x = x - np.max(x)
print(np.exp(x), '\n')
print(np.sum(np.exp(x)), '\n')
print(np.exp(x)/np.sum(np.exp(x)), '\n')

print('--------cross_entropy_error---------')








