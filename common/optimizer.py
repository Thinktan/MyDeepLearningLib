# coding: utf-8
import sys
sys.path.append('..')
from common.np import *

class SGD:
    '''
    随机梯度下降法(stochastic gradient descent)
    '''
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]