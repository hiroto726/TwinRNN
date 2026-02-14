# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 10:24:53 2024

@author: Fumiya
"""
# create custom initializer
import tensorflow as tf
from tensorflow.keras.initializers import Orthogonal

class OrthoCustom3(Orthogonal):
    def __init__(self, gain=1.0, seed=None,nUnit=2,nInh=1):
        super().__init__(gain=gain, seed=seed)
        self.nUnit=nUnit
        self.nInh=nInh
        self.conProb=0
 
    def initializer_ortho(self, shape,dtype=None):
        dtype = dtype or tf.float32
        w0=super().__call__(shape=shape,dtype=dtype)  
        # Create a mask for the diagonal elements
        mask = tf.eye(tf.shape(w0)[0], tf.shape(w0)[1], dtype=w0.dtype)
        wlog = w0 * (1 - mask)
        wloga=tf.math.abs(wlog[:-self.nInh,])
        wlogb=-tf.math.abs(-((1+self.conProb)*(self.nUnit-self.nInh)/self.nInh)*wlog[-self.nInh:,])
        return tf.cast(tf.concat([wloga, wlogb],0), dtype=w0.dtype)       
    
    def initializer_ortho_diag(self, shape,dtype=None):
        dtype = dtype or tf.float32
        w0=super().__call__(shape=shape,dtype=dtype)  
        # Create a mask for the diagonal elements
        wloga=tf.math.abs(w0[:-self.nInh,])
        wlogb=-tf.math.abs(-((self.nUnit-self.nInh)/self.nInh)*w0[-self.nInh:,])
        return tf.cast(tf.concat([wloga, wlogb],0), dtype=w0.dtype)   
 
    def __call__(self, shape,dtype=None):
        shape=[self.nUnit,self.nUnit]
        W= self.initializer_ortho(shape,dtype=None)
        W=W/(1+self.conProb)
        return W
    # eliminate diagonal element and 
        
    def get_config(self):
        return {"nUnit": self.nUnit, "nInh": self.nInh, "seed": self.seed,"gain":self.gain}        