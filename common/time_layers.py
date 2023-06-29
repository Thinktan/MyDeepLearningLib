# coding: utf-8
import sys
sys.path.append('..')
from common.np import *
from common.layers import SigmoidWithLoss, Embedding
from common.functions import softmax, sigmoid

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
            dx, dh = layer.backward(dhs[:, t, :] + dh) # 梯度为上方和右方传过来的梯度之和来计算
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

class LSTM:
    def __init__(self, Wx, Wh, b):
        '''

        Parameters
        ----------
        Wx: 输入'x'的全中参数(组合4个权重) shape: (D, 4H)
        Wh: 隐藏状态'h'的权重参数(组合4个权重) shape: (H, 4H)
        b: 偏差(组合4个偏差) shape: (4H,)
        '''
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        '''

        :param x:  shape -> (N, D)
        :param h_prev: shape -> (N, H)
        :param c_prev: shape -> (N, H)
        :return:
        '''
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        # (N,D)*(D,4H) + (N,H)*(H,4H) + (4H,) -> (N,4H)
        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next # shape -> (N,H)

    def backward(self, dh_next, dc_next):
        '''

        :param dh_next: shape -> (N,H)
        :param dc_next: shape -> (N,H)
        :return:
        '''
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        tanh_c_next = np.tanh(c_next)

        ds = dc_next + (dh_next * o) * (1 - tanh_c_next ** 2)

        dc_prev = ds * f

        di = ds * g
        df = ds * c_prev
        do = dh_next * tanh_c_next
        dg = ds * i

        di *= i * (1 - i)
        df *= f * (1 - f)
        do *= o * (1 - o)
        dg *= (1 - g ** 2)

        dA = np.hstack((df, dg, di, do))

        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev


class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h

            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0

        grads = [0, 0, 0] # dWx, dWh, db
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh
        return dxs

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None

class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W # shape -> (V,D), V: 词汇表数量, D: 向量维度

    def forward(self, xs):
        # N：mini-batch size
        # T: 时间步长
        N, T = xs.shape

        V, D = self.W.shape

        out = np.empty((N, T, D), dtype='f')
        self.layers = []

        # 按照时间步长处理，mini-batch: N
        for t in range(T):
            layer = Embedding(self.W)
            # out[:, t, :].shape -> (N,D)
            # xs[:, t].shape -> (N,)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad
        return None


class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        N, T, D = x.shape
        W, b = self.params # W: D*H, b: H*1

        rx = x.reshape(N*T, -1) # N,T,D -> N*T,D
        out = np.dot(rx, W) + b # out: N*T,H
        self.x = x
        return out.reshape(N, T, -1)

    def backward(self, dout):
        '''

        :param dout: shape: N,T,H
        :return:
        '''
        x = self.x # N,T,D
        N, T, D = x.shape
        W, b = self.params # W: D*H, b: H*1

        dout = dout.reshape(N*T, -1) # dout: N,T,H -> N*T,H
        rx = x.reshape(N*T, -1) # N,T,D -> N*T,D

        db = np.sum(dout, axis=0) # (H,)
        dW = np.dot(rx.T, dout) # dW.shape: (D,N*T)*(N*T,H) -> (D,H)
        dx = np.dot(dout, W.T) # dx.shape: (N*T,H)*(H,D) -> (N*T,D)
        dx = dx.reshape(*x.shape) # dx.shape: (N*T,D) -> (N,T,D)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx


class TimeSigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.xs_shape = None
        self.layers = None

    def forward(self, xs, ts):
        N, T = xs.shape
        self.xs_shape = xs.shape

        self.layers = []
        loss = 0

        for t in range(T):
            layer = SigmoidWithLoss()
            # xs[:,t].shape: (N,)
            # ts[:,t].shape: (N,)
            loss += layer.forward(xs[:, t], ts[:, t])
            self.layers.append(layer)

        return loss / T

    def backward(self, dout=1):
        N, T = self.xs_shape
        dxs = np.empty(self.xs_shape, dtype='f')

        dout *= 1/T
        for t in range(T):
            layer = self.layers[t]
            dxs[:, t] = layer.backward(dout)

        return dxs #dxs.shape: (N,T)

class TimeSoftmaxWithLoss:
    '''
    explain from gpt4.0
    这段代码定义了一个名为TimeSoftmaxWithLoss的类，
    它是一个用于处理序列数据的Softmax激活函数和损失计算的组合层。
    这个类有两个主要方法：forward和backward，分别用于前向传播和反向传播。
    mask是一个重要的概念，它用于处理不同长度的序列数据。在训练循环神经网络（RNN）时，
    我们通常需要处理不同长度的序列。为了将这些序列放入一个批次中，
    我们需要对较短的序列进行填充（padding），使它们与最长序列具有相同的长度。
    然而，在计算损失和梯度时，我们不希望这些填充值对结果产生影响。这就是mask的作用。
    mask是一个与ts（目标序列）形状相同的布尔数组，其中True表示对应位置的元素不是填充值（即有效数据），
    False表示对应位置的元素是填充值。在这个代码中，填充值由self.ignore_label表示，默认值为-1。
    在forward方法中，mask用于将填充值对应的损失设置为0，从而在计算总损失时不考虑这些值。
    在backward方法中，mask用于将填充值对应的梯度设置为0，从而在更新参数时不考虑这些值。
    总之，mask是一个用于处理不同长度序列数据的关键概念，它确保了填充值不会对损失和梯度产生影响。
    '''
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 如果监督标签是one-hot编码，转成索引格式
            ts = ts.argmax(axis=2)

        # 这段代码的作用是将ts中等于ignore_label的元素设置为False，其余元素设置为True，
        # 生成一个与ts形状相同的掩码数组mask。(N*T)
        mask = (ts != self.ignore_label)

        # 合并批次和时间序列N*T（reshape）
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # 在forward方法中，mask用于将填充值对应的损失设置为0，从而在计算总损失时不考虑这些值。
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        # explain from gpt4, it's nice~

        # 从缓存中获取之前在forward方法中存储的变量：
        # ts（目标序列），
        # ys（Softmax激活后的输出），
        # mask（用于处理填充值的布尔数组）
        # 和(N, T, V)（输入数据的形状参数）。
        ts, ys, mask, (N, T, V) = self.cache

        # 计算Softmax激活函数关于输入xs的梯度。首先，我们将ys（Softmax输出）赋值给dx。
        # 然后，我们使用NumPy的高级索引功能，将ts中的正确类别对应的梯度减1。
        # 这是因为Softmax损失函数关于正确类别的梯度是ys - 1，而对于其他类别的梯度是ys。
        dx = ys # shape -> (N*T,V)
        dx[np.arange(N * T), ts] -= 1 # shape -> (N*T,V)

        # 将dx乘以dout，这是损失函数关于Softmax层输出的梯度。在
        # 这个例子中，dout默认值为1，因为我们计算的是损失函数关于输入xs的直接梯度。
        dx *= dout

        # 将dx除以mask的总和（即有效数据的数量），这是为了对梯度进行归一化处理。
        # 这样做可以确保在不同批次大小和序列长度的情况下，梯度的大小保持一致。
        dx /= mask.sum()

        # 使用mask[:, np.newaxis]来将mask从形状(N*T,)扩展为形状(N*T,1)，
        # 以便将其与dx(N*T,V)相乘。这样做可以确保填充值对应的梯度设置为0。
        dx *= mask[:, np.newaxis]

        # 最后，将dx的形状从(N * T, V)调整为原始输入数据的形状(N, T, V)，
        # 以便在神经网络中继续进行反向传播。
        dx = dx.reshape((N, T, V))

        return dx

















