# coding: utf-8

from common.np import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)


def softmax(x):
    '''
    多类别分类
    :param x: 数组
    :return:
    mark: 减去max值原因 https://zhuanlan.zhihu.com/p/29376573
    '''
    # print('softmax x: ', x.shape)
    if x.ndim == 2:
        # learn from test.np/format1
        x = x - x.max(axis=1, keepdims=True)
        # print(type(x))
        # print(x.shape)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        # learn from test.np/format2
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x

# 损失函数：交叉熵函数
def cross_entropy_error(y, t):
    '''
    :param y: 1维或者2维数组，softmax输出
    :param t: 可以是2维的one-hot数组，或者是y每行最大值索引的数组
    :return: float 交叉熵
    mark: y为1维数组，t为index索引时，t最后是一个[[1]]格式，这不影响运算；其余情况为1维index数组
    '''
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size