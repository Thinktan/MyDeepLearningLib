# coding: utf-8
import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import Embedding, SigmoidWithLoss
import collections

# 负采样思想：将多分类问题转成二分类问题

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
        out = np.sum(target_W*h, axis=1) # (N,) 逐元素乘法操作 -> sum

        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        '''

        :param dout:  shape(1*N)
        :return:
        '''
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1) # (N,1)
        # dout之前属于sum操作，因此广播回形状，基于乘法的反向传播逻辑，计算dtarget_W和dh
        dtarget_W = dout * h # (N*1 -> N*D)x(N*D)=(N,D)
        self.embed.backward(dtarget_W)
        dh = dout*target_W #(N*1 -> N*D)x(N*D)=(N*D)

        return dh


class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        '''

        :param corpus: 单词ID列表
        :param power: 次方值
        :param sample_size: 负例的采样个数
        '''
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i] # i: word_id

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                p[target_idx] = 0 # 避免采样到自己
                p /= p.sum() # 因为target_idx位置设置了0，因此重新计算概率分布，保证总和为0
                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        else:
            # GPU模式下，速度也欧先，存在可能正例被当作反例的情况
            negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
                                                replace=True, p=self.word_p)

        return negative_sample


class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        '''

        :param W: 输出侧权重W
        :param corpus:  语料库(单词ID列表)corpus
        :param power: 概率分布的次方值power
        :param sample_size: 负例的采样数sample_size
        '''
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)

        # sample_size个负例+1个正例
        # loss_layer[0]和embed_dot_layers[0]用于处理正例
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        # 所有EmbeddingDot层共享输出层W矩阵
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        '''

        :param h: 隐藏层神经元 N*D, N: mini-batch
        :param target: 正例目标(N,) 一维数组
        :return:
        '''
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target) # 维度: (N,sample_size)

        # 正例的正向传播forward
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32) # (batch_size,)
        loss = self.loss_layers[0].forward(score, correct_label)

        # 负例的正向传播
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i] # 取出负例单词ID(negative_target)
            score = self.embed_dot_layers[1 + i].forward(h, negative_target) # 每个target得分
            loss += self.loss_layers[1 + i].forward(score, negative_label) # 每个target损失加起来

        return loss

    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)

        return dh