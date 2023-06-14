# coding: utf-8

from common.np import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)


def softmax(x):
    '''
    多类别分类
    :param x:
    :return:
    '''
    if x.ndim == 2:
        # learn from test.np/format1
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        # learn from test.np/format2
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x

# 损失函数：交叉熵函数
