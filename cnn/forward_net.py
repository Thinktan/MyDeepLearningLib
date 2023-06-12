# coding: utf-8
import numpy as np

# 层的类化及正向传播的实现
class Sigmoid:
    def __init__(self):
        self.params = []

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

# 隐藏层计算
# h = xW + b
# x: 1x2
# W: 2x4
# b: 1x4
# h: 1x4

# x: Nx2(一个输入是一行)
# W: 2x4(一个神经元的参数在一列，列数是神经元个数)
# b: 1x4(boardcast -> Nx4)(列数是神经元个数)
# h: Nx4

class Affine:
    def _init__(self, W, b):
        self.params = [W, b] # array

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# test code
np.random.seed(1234)

x = np.random.randn(10, 2) # 输入

W1 = np.random.randn(2, 4) # 权重
b1 = np.random.randn(4)    # 偏置

W2 = np.random.randn(4, 3)
b2 = np.random.randn(3)

h = np.dot(x, W1)+b1
a = sigmoid(h)
s = np.dot(a, W2) + b2

print(h.shape)
print(a.shape)
print(s.shape)

print('------------------------')

print(W1.shape)
print(b1.shape)
params = [W1, b1]
print(params)

print('------------------------')
x = np.arange(0, 4, 1)
print(x)
print(sigmoid(x))











