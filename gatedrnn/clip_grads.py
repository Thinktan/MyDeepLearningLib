import sys
sys.path.append('..')
import os
from common.np import *
from common.util import clip_grads

rn = np.random.rand

dW1 = rn(3, 3)*10
dW2 = rn(3, 3)*10

grads = [dW1, dW2]

max_norm = [dW1, dW2]
max_norm = 5.0

print(grads, '\n')
clip_grads(grads, max_norm)
print(grads, '\n')