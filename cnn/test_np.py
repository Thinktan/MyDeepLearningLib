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
x = np.array([[1, 5, 3]])
print(np.max(x, keepdims=True))
print(x - np.max(x))
x = x - np.max(x)
print(np.exp(x))
print(np.sum(np.exp(x)))
print(np.exp(x)/np.sum(np.exp(x)))

print('--------cross_entropy_error---------')
x = np.array([1, 4, 3])
# print(x.ndim)
# print(x.reshape(1, x.size).ndim)
t = np.array([[1, 3, 6], [6, 5, 4]])
# print(np.sum(t))

print('t.argmax: ', t.argmax(axis=1))
xy = x.reshape(1, x.size)
print('xy.argmax: ', xy.argmax(axis=1))
print('--------------')
# y: 1 dim vec, t: index
xt = np.array([1]).reshape(1, np.array([1]).size)
print('xt: ', xt)
print('a1: ', xy[np.arange(xy.shape[0]), xt])
print('b1: ', np.log(  xy[np.arange(xy.shape[0]), xt] ))
print('c1: ', -np.sum( np.log(  xy[np.arange(xy.shape[0]), xt] ) ) )
print('d1: ', -np.sum( np.log(  xy[np.arange(xy.shape[0]), xt] ) ) / t.shape[0])



print('--------------')
# y: multi dim vec, t: multi one hot vec or multi index
print('a1: ', t[np.arange(t.shape[0]), [2, 1]])
print('a2: ', t[np.arange(t.shape[0]), [[2, 1]]])
print('b1: ', np.log( t[np.arange(t.shape[0]), [2, 1]] ))
print('b2: ', np.log( t[np.arange(t.shape[0]), [[2, 1]]] ))
# print(np.log(2))
print('c1: ', -np.sum( np.log( t[np.arange(t.shape[0]), [2, 1]] )) )
print('c2: ', -np.sum( np.log( t[np.arange(t.shape[0]), [[2, 1]]] )) )
print('d1: ', -np.sum( np.log( t[np.arange(t.shape[0]), [2, 1]] )) / t.shape[0])
print('d2: ', -np.sum( np.log( t[np.arange(t.shape[0]), [[2, 1]]] )) / t.shape[0])





