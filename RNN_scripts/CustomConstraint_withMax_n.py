# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 00:53:26 2024

@author: Fumiya
"""
import tensorflow as tf
from tensorflow.keras.constraints import Constraint

class IEWeightandLim_n(Constraint):
    def __init__(self, nUnit=1, nInh=0, W_log=None, W_offdiag=None, RNN_num=1,maxval=None):
        self.nUnit=nUnit
        self.nInh=nInh
        self.W_log=W_log # matrix to set certain element 0
        self.maxval=maxval
        self.RNN_num=RNN_num
        self.W_offdiag=W_offdiag # binary matrix where off diagonal blocks are 1 and 0 otherwise        
    
        
    def __call__(self, w, dtype=None):
        shape=[self.nUnit*self.RNN_num, self.nUnit*self.RNN_num]
        dtype = dtype or tf.float32
        w=tf.cast(w,dtype=dtype)
        wloga=tf.ones([self.nUnit-self.nInh, shape[1]],dtype=dtype)
        wlogb=tf.zeros([self.nInh, shape[1]],dtype=dtype)
        wlog2=tf.concat([wloga, wlogb],0)
        wlog=tf.tile(wlog2,[self.RNN_num,1]) # this creates a matrix where excitory inputs are 1 and the rest is 0  
        w0= tf.constant([0],dtype=dtype)
        
        w1 = tf.cast(tf.math.maximum(w * wlog, w0), dtype=dtype) # matrix where positive value of excitory elements are taken
        w2 = tf.cast(tf.math.minimum(w * (1 - wlog), w0), dtype=dtype) # matrix where negative value of inhibitory elements are taken
        W=w1+w2 # make excitory inputs positive and inhibitory negative

        #constrain with max value of off diagonal blocks
        if self.maxval is not None:
            W_off=tf.math.minimum(W*self.W_offdiag, self.maxval) 
        else:
            W_off=W*self.W_offdiag 
        W=W*(1-self.W_offdiag)+W_off*self.W_offdiag #replace off diagonal block with W_off
        
        
        
        # remove diagonal element
        mask = tf.eye(shape[0], shape[1], dtype=dtype)
        W = W * (1 - mask)

        # make some element zero to make connection sparse and make inter inhibitory connection 0
        W=self.W_log*W
        
        return W


    def get_config(self):
        return {
            "nUnit":self.nUnit,
            "nInh": self.nInh,
            "W_log": self.W_log,  # Corrected "A_mask" to "W_log"
            "W_offdiag": self.W_offdiag,  # Corrected "B_mask" to "W_offdiag"
            "maxval": self.maxval,
            "RNN_num": self.RNN_num  # Added RNN_num to the config
        }      

class IEWeightOut_n(Constraint):
    def __init__(self, nInh=0):
        self.nInh=nInh
    

    def __call__(self, w):
        w=tf.convert_to_tensor(w)
        w0= tf.constant([0],dtype=w.dtype)
        w1=tf.math.greater_equal(w[:-self.nInh,], w0)
        w2=tf.math.less_equal(w[-self.nInh:,], w0)
        return w*tf.cast(tf.concat([w1,w2], 0), dtype=w.dtype)# no need to eliminate diagonal element since its used for ouput

    def get_config(self):
        return {"nInh": self.nInh}   
# use the code below to check if the custom constraint is indeed working
#nInh=1
#weight = tf.convert_to_tensor((-1.0, 1.0))
#with tf.compat.v1.Session() as sess:
#    rnd = sess.run(IEWeight(nInh=nInh)(weight))
#print(rnd)