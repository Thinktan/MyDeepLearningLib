import sys
sys.path.append('..')

from common.np import *
from common.time_layers import *

class Encoder:
    def __init(self, vocab_size, wordvec_size, hidden_size):
        '''

        :param vocab_size: 词汇表数量(数字0-9，"+", "", "_")
        :param wordvec_size: 字符向量维度
        :param hidden_size: LSTM隐藏状态维度
        :return:
        '''
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)

        self.params = self.embed.params + self.lstm.params
        self.grads = self.embed.grads + self.lstm.grads
        self.hs = None

    def forward(self, xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        self.hs = hs
        return hs[:, -1, :] # 取最后一个时刻的隐藏状态

    def backward(self, dh):
        dhs = np.zeros_like(self.hs)
        dhs[:, -1, :] = dh # 将dh放入到dhs中对应位置，即最后一个时刻位置

        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout