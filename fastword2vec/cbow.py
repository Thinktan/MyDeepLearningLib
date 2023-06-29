
import sys
sys.path.append('..')
from common.np import *
from common.layers import Embedding
from fastword2vec.negative_sampling_layer import NegativeSamplingLoss

class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        '''

        :param vocab_size: 词汇量
        :param hidden_size: 中间层的神经元个数
        :param window_size:上下文的大小
        :param corpus:单词ID列表
        '''

        V, H = vocab_size, hidden_size

        # 初始化权重
        W_in = 0.01*np.random.randn(V, H).astype('f')
        W_out = 0.01*np.random.randn(V, H).astype('f')

        # 生成层
        self.in_layers = []
        for i in range(2*window_size):
            layer = Embedding(W_in)
            self.in_layers.append(layer)

        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        # 将所有权重和梯度整理到列表中
        layers = self.layer + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 将单词的分布式表示设置为成员变量
        self.word_vecs = W_in

    def forward(self, contexts, target):
        '''

        :param contexts: 二维数组，单词ID形式
        :param target: 一维数组，单词ID形式
        :return:
        '''
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts)

        h *= 1/len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1/len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)

        return None