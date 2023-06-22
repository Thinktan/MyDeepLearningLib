import sys
sys.path.append('..')
from negative_sampling_layer import UnigramSampler
from common.np import *

corpus = np.array([0, 1, 2, 3, 4, 1, 2, 3])
power = 0.75
sample_size = 2
sampler = UnigramSampler(corpus, power, sample_size)
target = np.array([1, 3, 0])
negative_sample = sampler.get_negative_sample(target)
print(negative_sample)

correct_label = np.ones(11, dtype=np.int32)
print(correct_label.shape)