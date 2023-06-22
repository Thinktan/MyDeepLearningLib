# coding: utf-8
import sys

import numpy as np

sys.path.append('..')
from common.config import GPU
from common.functions import softmax, cross_entropy_error, relu
#import numpy as np
from common.np import *

# 层的类化及正向传播的实现
class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0-self.out) * self.out
        return dx

class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoid输出
        self.t = None  #

    def forward(self, x, t):
        '''
        :param x: N*1 N(mini-batch)
        :param t: N*1 or N,
        :return: loss
        在 `SigmoidWithLoss` 类的 `forward` 方法中，`np.c_` 是用来将两个数组沿第二个轴（列方向）连接起来的。
        这是因为在这个情况下，计算交叉熵损失函数需要预测的概率分布，而不仅仅是正类别的概率。
        对于二分类问题，sigmoid 函数的输出 `self.y` 是正类别（类别 1）的概率。
        对于负类别（类别 0）的概率，可以通过 `1 - self.y` 来计算。
        因此，`np.c_[1 - self.y, self.y]` 产生了一个形状为 `(batch_size, 2)` 的数组，
        其中每一行都包含一个样本的负类别和正类别的概率。这个概率分布可以直接传递给 `cross_entropy_error` 函数，
        以计算交叉熵损失。
        总的来说，`np.c_` 的使用使得这个函数可以处理一个完整的概率分布，而不仅仅是一个类别的概率。
        '''
        self.t = t
        self.y = 1 / (1 + np.exp(-x)) # N*H
        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) * dout / batch_size
        return dx

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
    W: D*H
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
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


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


    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx/batch_size

        return dx

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

class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out


    def backward(self, dout):
        dW, = self.grads # 别名，实际是引用
        dW[...] = 0

        # for i, word_id in enumerate(self.idx):
        #     dW[word_id] += dout[i]
        # 等同于np.add.at(dW, self.idx, dout) 参考fastword2vec/test_np_add.py

        if GPU:
            np.scatter_add(dW, self.idx, dout)
        else:
            np.add.at(dW, self.idx, dout)

        return None

class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.params.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        '''前向传播

        :param h: (N,D) 中间层神经元 D:神经元个数, W形状是：(w, D)
        :param idx: (N,) 单词ID列表 mini-batch，从W(w,D)中取出N个得到(N,D)
        :return: (N,)
        '''
        target_W = self.embed.forward(idx) # (N,D)
        out = np.sum(target_W*h, axis=1) # (N,) dot(矩阵乘法) -> sum

        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        '''

        :param dout:  shape(1*N)
        :return:
        '''
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1) # (N,1)
        # dout之前属于sum操作，因此广播回形状，后面两个d运算是MatMul操作
        dtarget_W = dout * h # (N*1 -> N*D)x(N*D)=(N,D)
        self.embed.backward(dtarget_W)
        dh = dout*target_W #(N*1 -> N*D)x(N*D)=(N*D)

        return dh






