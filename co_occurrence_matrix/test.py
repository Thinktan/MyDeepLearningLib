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

word_to_id = {}
id_to_word = {}

for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word

print(word_to_id, '\n')
print(id_to_word, '\n')

corpus = [word_to_id[w] for w in words]
corpus = np.array(corpus)
print(corpus)

x = np.array([[1, 2, 3]])
y = np.array([[5], [6], [7]])
print(np.dot(x, y))
print(5+12+21)
print(x*y)

x = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([[2, 3], [3, 1]])
print(x)
print(y)
print(np.dot(x, y))
