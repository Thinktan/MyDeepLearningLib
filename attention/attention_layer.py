# coding: utf-8
import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import Softmax


class WeightSum:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, hs, a):
        '''

        :param hs: shape(N,T,H)
        :param a: shape(N,T)
        :return: shape(N,H)
        '''
        N, T, H = hs.shape # shape(N,T,H)

        ar = a.reshape(N, T, 1).repeat(H, axis=2) # reshape (N,T) -> (N,T,1), repeated -> (N,T,H)
        t = hs * ar # shape(N,T,H)
        c = np.sum(t, axis=1) # shape(N,H)

        self.cache = (hs, ar)
        return c

    def backward(self, dc):
        '''

        :param dc: shape(N,H)
        :return: dhs.shape(N,T,H),da.shape(N,T)
        '''
        hs, ar = self.cache
        N, T, H = hs.shape
        dt = dc.reshape(N, 1, H).repeat(T, axis=1)
        dar = dt * hs
        dhs = dt * ar
        da = np.sum(dar, axis=2)

        return dhs, da


class AttentionWeight:
    def __init__(self):
        self.params, self.grads = [], []
        self.softmax = Softmax()
        self.cache = None

    def forward(self, hs, h):
        '''

        :param hs: shape(N,T,H)
        :param h: shape(N,H)
        :return: a.shape(N,T)
        '''
        N, T, H = hs.shape

        hr = h.reshape(N, 1, H).repeat(T, axis=1) # (N,T,H)
        t = hs * hr # (N,T,H)
        s = np.sum(t, axis=2) # (N,T)
        a = self.softmax.forward(s) # (N,T)

        self.cache = (hs, hr)
        return a # (N,T)

    def backward(self, da):
        '''

        :param da: shape(N,T)
        :return: dhs.shape(N,T,H), dh.shape(N,H)
        '''
        hs, hr = self.cache
        N, T, H = hs.shape

        ds = self.softmax.backward(da)
        dt = ds.reshape(N, T, 1).repeat(H, axis=2) # (N,T,H)
        dhs = dt * hr
        dhr = dt * hs
        dh = np.sum(dhr, axis=1) #(N,H)

        return dhs, dh


class Attention:
    def __init__(self):
        self.params, self.grads = [], []
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()
        self.attention_weight = None

    def forward(self, hs, h):
        '''

        :param hs: shape(N,T,H)
        :param h: shape(N,H)
        :return: out.shape(N,H)
        '''
        a = self.attention_weight_layer.forward(hs, h) # hs.shape(N,T,H), h.shape(N,H) -> a.shape(N,T)
        out = self.weight_sum_layer.forward(hs, a) # hs.shape(N,T,H), a.shape(N,T) -> out.shape(N,H)
        self.attention_weight = a # shape(N,T)
        return out

    def backward(self, dout):
        '''

        :param dout: shape(N,H)
        :return: dhs.shape(N,T,H), dh.shape(N,H)
        '''
        dhs0, da = self.weight_sum_layer.backward(dout)
        dhs1, dh = self.attention_weight_layer.backward(da)
        dhs = dhs0 + dhs1
        return dhs, dh

class TimeAttention:
    def __init__(self):
        self.params, self.grads = [], []
        self.layers = None
        self.attention_weights = None

    def forward(self, hs_enc, hs_dec):
        '''
        mark: 上下场景中，T指代encoder侧的输入长度，Td表示decoder侧时序长度
        :param hs_enc: shape(N,T,H)
        :param hs_dec: shape(N,Td,H)
        :return: out.shape(N,Td,H)
        '''
        N, Td, H = hs_dec.shape
        out = np.empty_like(hs_dec)
        self.layers = []
        self.attention_weights = []

        for t in range(Td):
            layer = Attention()
            # hs_enc.shape(N,T,H), hs_dec[:,t,:].shape(N,H)
            out[:, t, :] = layer.forward(hs_enc, hs_dec[:,t,:])
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)

        return out

    def backward(self, dout):
        N, T, H = dout.shape
        dhs_enc = 0
        dhs_dec = np.empty_like(dout)

        for t in range(T):
            layer = self.layers[t]
            dhs, dh = layer.backward(dout[:, t, :])
            dhs_enc += dhs
            dhs_dec[:,t,:] = dh

        return dhs_enc, dhs_dec