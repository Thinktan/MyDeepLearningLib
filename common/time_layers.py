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