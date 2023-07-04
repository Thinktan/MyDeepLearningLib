
import sys
sys.path.append('..')
from common.time_layers import *
from seq2seq.seq2seq import Encoder, Seq2seq
from attention.attention_layer import TimeAttention

# 编码器
# 改进1：将编码器的全部时刻的隐藏状态取出来

# 解码器
# 改进1：使用Attention层


class AttentionEncoder(Encoder):
    def forward(self, xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        return hs

    def backward(self, dhs):
        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout



class AttentionDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # embedding层参数同seq2seq
        embed_W = (rn(V, D) / 100).astype('f')

        # lstm层参数同seq2seq
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')

        # affine层略有区别，入参宽度是H*2，将Lstm层和attention层参数拼接起来
        affine_W = (rn(2*H, V) / np.sqrt(2*H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.attention = TimeAttention()
        self.affine = TimeAffine(affine_W, affine_b)
        layers = [self.embed, self.lstm, self.attention, self.affine]

        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, enc_hs):
        '''

        :param xs: shape(N,Td)
        :param enc_hs: shape(N,T,H)
        :return:
        '''
        h = enc_hs[:,-1]  # 取出最后一个时刻的输出，(N,T,H) -> shape(N,H)
        self.lstm.set_state(h)

        out = self.embed.forward(xs)  # out.shape: (N,Td,D)
        dec_hs = self.lstm.forward(out) # dec_hs.shape: (N,Td,H)
        c = self.attention.forward(enc_hs, dec_hs) # enc_hs:(N,T,H), dec_hs(N,Td,H), c(N,Td,H)
        out = np.concatenate((c, dec_hs), axis=2) # out:(N,Td,2H)
        score = self.affine.forward(out) # score: (N,Td,V)

        return score

    def backward(self, dscore):
        dout = self.affine.backward(dscore) # dout(N,Td,2H)
        N, T, H2 = dout.shape # T is Td
        H = H2 // 2

        dc, ddec_hs0 = dout[:,:,:H], dout[:,:,H:] #dc.shape(N,Td,H), dout.shape(N,Td,H)
        denc_hs, ddec_hs1 = self.attention.backward(dc) # denc_hs.shape(N,T,H), ddec_hs1.shape(N,Td,H)
        ddec_hs = ddec_hs0 + ddec_hs1 # ddec_hs.shape(N,Td,H)
        dout = self.lstm.backward(ddec_hs) # dout.shape(N,Td,D)
        dh = self.lstm.dh # (N,H)
        denc_hs[:, -1] += dh # (N,T,H)
        self.embed.backward(dout)

        return denc_hs # (N,T,H)

    def generate(self, enc_hs, start_id, sample_size):
        '''

        :param enc_hs: (N,T,H)
        :param start_id:
        :param sample_size:
        :return:
        '''
        sampled = []
        sample_id = start_id
        h = enc_hs[:, -1] # (N,H)
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array([sample_id]).reshape((1, 1))

            out = self.embed.forward(x)
            dec_hs = self.lstm.forward(out)
            c = self.attention.forward(enc_hs, dec_hs)
            out = np.concatenate((c, dec_hs), axis=2)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(sample_id)

        return sampled


class AttentionSeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        args = vocab_size, wordvec_size, hidden_size
        self.encoder = AttentionEncoder(*args)
        self.decoder = AttentionDecoder(*args)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
        self.grads = self.encoder.grads + self.decoder.grads