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
    def __init__(self, W, b):
        self.params = [W, b] # array

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        """
        :param input_size:每行输入的维度
        :param hidden_size:隐藏神经元的个数
        :param output_size:输出神经元的个数
        """

        I, H, O = input_size, hidden_size, output_size

        # weights and bias
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)

        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # W1 = np.arange(I*H).reshape(I, H)
        # b1 = np.arange(0, H, 1)
        #
        # W2 = np.arange(H*O).reshape(H, O)
        # b2 = np.arange(0, O, 1)

        # layer
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        # params list
        self.params = []
        # index = 0
        for layer in self.layers:
            # 将param拼接起来, np.array是引用类型
            self.params += layer.params
            # print("--------index:", index, '---------')
            # print('layer.params:\n', layer.params)
            # print('self.params:\n', self.params, '\n')
            # index += 1
        # print(len(self.params), '---------====') ans: 4 affine的W1,b1,W2,b2

    def predict(self, x):
        """
        :param x: N*I，N is batch size, I: input dimension
        :return:forward anser
        """
        for layer in self.layers:
            x = layer.forward(x)

        return x



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# test code
np.random.seed(1234)

x = np.random.randn(10, 2) # 输入

# W1 = np.random.randn(2, 4) # 权重
# b1 = np.random.randn(4)    # 偏置
#
# W2 = np.random.randn(4, 3)
# b2 = np.random.randn(3)

W1 = np.arange(8).reshape(2, 4)
b1 = np.arange(0, 4, 1)

W2 = np.arange(12).reshape(4, 3)
b2 = np.arange(0, 3, 1)



h = np.dot(x, W1)+b1
a = sigmoid(h)
s = np.dot(a, W2) + b2

print(h.shape)
print(a.shape)
print(s.shape)

print('------------------------')

x = np.random.randn(10, 2)
model = TwoLayerNet(2, 4, 3)
s = model.predict(x)
print(x)
print(s)

print('------------------------')

# mark:
# 在Python中，两个列表（list）相加会将他们的元素（不论是否是数组）连接在一起，形成一个新的、更长的列表，
# 而不会进行数组的拼接或广播相加。这是因为Python的列表并不具备NumPy数组那样的向量化运算特性。
# 如果你的列表元素是NumPy数组，并且你想对他们进行拼接操作，使用np.concatenate


# 假设有两个列表，元素是NumPy数组
list1 = [np.array([1, 2]), np.array([3, 4])]
list2 = [np.array([5, 6]), np.array([7, 8])]

# 使用numpy.concatenate将数组拼接在一起
concatenated = np.concatenate((list1[0], list2[0]))
print(concatenated)  # 输出：[1 2 5 6]
print(list1 + list2)


print('------------------------')

x = [np.arange(8).reshape(2, 4), np.arange(0, 4, 1)]
print(x)
print(type(x))

y = [np.arange(12).reshape(3, 4), np.arange(0, 6, 1)]
z = []
z += x
z += y
print(z)
print(len(z))
x[0][0][1] = 22
print(z)
x[0][1] = 55
print(z)