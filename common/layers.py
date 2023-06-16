import numpy as np

# coding: utf-8
from common.np import *  # import numpy as np
from common.config import GPU
from functions import softmax, cross_entropy_error, relu

# 层的类化及正向传播的实现
class Sigmoid:
    def __init__(self):
        self.params = []
        self.out = None

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
    '''
    shape:
    x: N*D
    W: N*H
    b: H --repeated--> N*H
    out(z): N*H
    '''
    def __init__(self, W, b):
        self.params = [W, b] # array
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None



    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class Softmax:
    def __init__(self):
        self.params = []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

class SoftmaxWithLoss:
    def __init__(self):
        self.params = []
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 转成索引形式
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss


class MatMul:
    def __init__(self, W):
        '''
        :param W: 参数
        维度：
        x: N*D(N: mini-batch cnt, D 维度)
        W: D*H
        out: N*H
        '''
        self.params = [W]
        self.grads = [np.zeros_like(W)] # 保存W的梯度
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx