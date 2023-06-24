# coding: utf-8
import sys
sys.path.append('..')
import os
from common.np import *
import numpy

def preprocess(text):
    '''预处理文本
    :param text:
    :return:
    '''
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word

def create_co_matrix(corpus, vocab_size, window_size=1):
    '''创建共现矩阵

    :param corpus: 单词ID列表
    :param vocab_size: 单词数
    :param window_size:窗口大小（当窗口大小为1时，上下文为单词左右各一个单词）
    :return: 共现矩阵
    '''
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix

def cos_similarity(x, y, eps=1e-8):
    ''' 余弦相似度计算

    :param x: 向量
    :param y: 向量
    :param eps: 防止向量全为0值导致分母为0
    :return: value
    '''
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)

def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    '''搜索相似词

    :param query: 查询词
    :param word_to_id: 单词到单词ID的字典
    :param id_to_word: 单词ID到单词的字典
    :param word_matrix: 汇总了单词向量的矩阵
    :param top: 显示到前几位
    '''
    # 1. 取出查询词 query_vec
    if query not in word_to_id:
        print('%s is not found' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 计算cos similarity
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size) # index: word_id, value: similarity
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # 基于余弦相似度，按降序输出值
    count = 0
    for i in (-1 * similarity).argsort():
        # i: index == word_id
        if id_to_word[i] == query:
            continue
        print('%u, %s: %s' % (count+1, id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return

def ppmi(C, verbose=False, eps = 1e-8):
    '''PPMI 创建点互信息 Pointwise Mutual information, PMI

    :param C: 共现矩阵
    :param verbose: 是否输出调试信息
    :return: PPMI
    '''
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C) # 全部元素相加
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j]*N/(S[j]*S[i] + eps))
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100+1) == 0:
                    print('%.1f%% done' % (100*cnt/total))

    return M

def create_contexts_target(corpus, window_size=1):
    '''根据语料库（单词ID列表）生成上下文与目标

    :param corpus:语料库（单词ID列表）
    :param window_size: 窗口大小
    :return: 上下文 + 目标
    '''
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)


def convert_one_hot(corpus, vocab_size):
    '''one-hot编码转换

    :param corpus: 单词ID列表，一维或者二维NumPy数组
    :param vocab_size: 单词数
    :return: one-hot编码，二维或者三维NumPy数组
    '''
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot


def to_cpu(x):
    import numpy
    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)


def to_gpu(x):
    import cupy
    if type(x) == cupy.ndarray:
        return x
    return cupy.asarray(x)
