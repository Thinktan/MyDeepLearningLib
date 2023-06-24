# coding: utf-8
import sys
sys.path.append('..')
from common.np import *  # import numpy as np

import pickle
from common.trainer import  Trainer
from common.optimizer import Adam
from fastword2vec.cbow import CBOW
from common.util import create_contexts_target, to_cpu
from dataset import ptb
from common.config import GPU

# 设定超参数
window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10

# 读入数据
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)
# if GPU:
#     contexts, target = to_gpu(contexts), to_gpu(target)

# 生成模型
model = CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# 开始学习
trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

# 保存数据
word_vecs = model.word_vecs
if GPU:
    word_vecs = to_cpu(word_vecs)

params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
pkl_file = 'my_cbow_params.pkl'

with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)


