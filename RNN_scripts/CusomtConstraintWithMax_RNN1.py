
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 00:53:26 2024

@author: Fumiya
"""
import tensorflow as tf
from tensorflow.keras.constraints import Constraint

class IEWeightandLim(Constraint):
    def __init__(self, nInh=0, maxval=1):
        self.nInh=nInh
        self.maxval=maxval
        
    def constraint_w(self, w):
        w=tf.convert_to_tensor(w)
        w0= tf.constant([0],dtype=w.dtype)
#        w[:-self.nInh,]=tf.math.maximum(w0, w[:-self.nInh,])
#        w[-self.nInh:,]=tf.math.minimum(w0,w[-self.nInh:,])
        w1=tf.math.greater_equal(w[:-self.nInh,], w0)
        w2=tf.math.less_equal(w[-self.nInh:,], w0)
        return (w-w*tf.eye(tf.shape(w)[0], num_columns=tf.shape(w)[1]))*tf.cast(tf.concat([w1,w2], 0), dtype=w.dtype) # explicitly eliminate diagonal element
    
    def constraint_diag(self, w):
        w=tf.convert_to_tensor(w)
        w0= tf.constant([0],dtype=w.dtype)
        w1=tf.math.greater_equal(w[:-self.nInh,], w0)
        w2=tf.math.less_equal(w[-self.nInh:,], w0)
        return w*tf.cast(tf.concat([w1,w2], 0), dtype=w.dtype) 
    
        
    def __call__(self, w):
        w=self.constraint_w(w)
        return w

    

class IEWeightOut(Constraint):
    def __init__(self, nInh=0):
        self.nInh=nInh
    

    def __call__(self, w):
        w=tf.convert_to_tensor(w)
        w0= tf.constant([0],dtype=w.dtype)
#        w[:-self.nInh,]=tf.math.maximum(w0, w[:-self.nInh,])
#        w[-self.nInh:,]=tf.math.minimum(w0,w[-self.nInh:,])
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