# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 00:53:26 2024

@author: Fumiya
"""
import tensorflow as tf
from tensorflow.keras.constraints import Constraint

class IEWeightandLim_same(Constraint):
    def __init__(self, nInh=0, A_mask=1,B_mask=1,maxval=1):
        self.nInh=nInh
        self.A_mask=A_mask
        self.B_mask=B_mask
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
        W_A, S_A=tf.split(w,num_or_size_splits=2, axis=1)
        W_A=self.constraint_w(W_A)
        S_A=self.constraint_diag(S_A)
        S_A=S_A*tf.cast(self.A_mask,dtype=S_A.dtype)
        S_A=tf.math.minimum(S_A,self.maxval)        
        
        W=tf.concat([W_A,S_A],1)
        return W


    def get_config(self):
        return {"nInh": self.nInh,
                "A_mask":self.A_mask,
                "B_mask":self.B_mask,
                "maxval":self.maxval}      

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