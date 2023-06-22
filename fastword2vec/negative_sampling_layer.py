# coding: utf-8
import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import EmbeddingDot, SigmoidWithLoss
import collections

# 负采样思想：将多分类问题转成二分类问题

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