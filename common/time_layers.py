# coding: utf-8
import sys
sys.path.append('..')
from common.np import *

class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        '''

        :param x:
        :param h_prev:
        :return:
        '''
        Wx, Wh, b = self.params
        # 形状检查
        # h_prev: N*H, Wh: H*H, h_prev*Wh -> N*H
        # x: N*D, Wx: D*H, x*Wx -> N*H
        # h_next: N*H
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next ** 2)
        db = np.sum(dt, axis=0)
        dWh = np.dot(h_prev.T, dt)
        dh_prev = np.dot(dt, Wh.T)
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev

class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        # h保存调用forward方法时的最后一个RNN层的隐藏状态
        # dh保存调用backward方法时，传给前一个块的隐藏状态的梯度
        self.h, self.dh = None, None

        # stateful: 是否保存中间
        self.stateful = stateful

    def forward(self, xs):
        '''

        :param xs: 形状 N*T*D
        :return: hs: 形状 N*T*H
        '''
        Wx, Wh, b = self.params
        N, T, D = xs.shape # N个mini-batch，T个时间步长，D为输入向量维度
        D, H = Wx.shape # H是隐藏状态向量维度

        self.layers = []
        # hs: 输出容器，用于存放每个RNN层的输出
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)  # xs[:, t, :] -> 格式为N*H
            hs[:, t, :] = self.h # N*H
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        '''

        :param dhs: N*T*H
        :return: dxs: N*T*D
        '''
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype='f') # 给下一层的梯度
        dh = 0 # 给左侧块的梯度

        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        self.dh = dh

        return dxs



    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None
























