# coding: utf-8

import sys
sys.path.append('..')
from common.np import *

text = 'You say goodbye and I say hello.'

text = text.lower()
text = text.replace('.', ' .')
print(text)

words = text.split(' ')
print(words, '\n')