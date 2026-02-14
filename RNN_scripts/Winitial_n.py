# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 10:24:53 2024

@author: Hiroto
"""
# create custom initializer
import tensorflow as tf
from tensorflow.keras.initializers import Orthogonal


# initialize recurrent matrix for n RNNs. W_log specifies zero index
class OrthoCustom_n(Orthogonal):
    def __init__(self, gain=1.0, RNN_num=1,seed=None,nUnit=2,nInh=1,W_log=0,conProb=0):
        super().__init__(gain=gain, seed=seed)
        self.nUnit=nUnit
        self.RNN_num=RNN_num
        self.nInh=nInh
        self.Ex=self.nUnit-self.nInh
        self.W_log=W_log
        self.conProb=conProb
 
 
    def __call__(self, shape,dtype=None):
        shape=[self.nUnit*self.RNN_num, self.nUnit*self.RNN_num]
        dtype = dtype or tf.float32
        W_raw=super().__call__(shape=shape,dtype=dtype) 
        wloga=tf.ones([self.nUnit-self.nInh, shape[1]],dtype=dtype)
        wlogb=tf.zeros([self.nInh, shape[1]],dtype=dtype)
        wlog2=tf.concat([wloga, wlogb],0)
        wlog=tf.tile(wlog2,[self.RNN_num,1]) # this creates a matrix where excitory inputs are 1 and the rest is 0
        
        # coefficient to multiply inhibitory connection: aim to make net output 0
        coeff=(1+(self.RNN_num-1)*self.conProb)*((self.nUnit-self.nInh)/self.nInh)
        # convert indices with excitory input to positive and negative otherwise
        W=tf.math.abs(W_raw*wlog)-coeff*tf.math.abs(W_raw*(1-wlog))
        
        # remove diagonal element-> create mask first
        mask = tf.eye(shape[0], shape[1], dtype=dtype)
        W = W * (1 - mask)

        # make some element zero to make connection sparse and make inter inhibitory connection 0
        W=self.W_log*W
        return W
    # eliminate diagonal element and 
        
    def get_config(self):
        return {"nUnit": self.nUnit, 
                "RNN_num": self.RNN_num ,
                "Ex": self.Ex,
                "nInh": self.nInh, 
                "W_log": self.W_log,
                "conProb": self.conProb,
                "seed": self.seed,
                "gain":self.gain}        