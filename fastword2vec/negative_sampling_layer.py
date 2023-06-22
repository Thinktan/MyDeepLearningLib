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