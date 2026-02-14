# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 21:53:11 2024

@author: Hiroto Imamura
"""



import tensorflow as tf
from tensorflow.keras.layers import Layer

class QRDcell(Layer):
    def __init__(self, start=0,nUnit=0, **kwargs):
        super(QRDcell, self).__init__(**kwargs)
        self.start = start
        self.nUnit=nUnit



    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # Initial state for Q: identity matrix
        initial_Q = tf.linalg.eye(self.nUnit, batch_shape=[batch_size], dtype=dtype)
        # Initial state for iteration: zeros
        initial_iteration = tf.zeros((batch_size, 1), dtype=tf.int32)
        return [initial_Q, initial_iteration]


    @property
    def state_size(self):
        return [[self.nUnit, self.nUnit], 1]
    @tf.function
    def call(self, inputs, states):
        # inputs is J_{n} (batch, nUnit, nUnit)
        # states is Q_{n-1} (batch, nUnit, nUnit). Q_{0} is identity matrix
        # Q_{n}R_{n} = J_{n}Q_{n-1}
        # Q_{n} is the next state. (Q_new)
        # R_{n} is the output. (R_new)
        
        J = inputs
        Q = states[0]
        iteration = states[1]
        
        if iteration[0] >= self.start:
            Q_new, R_new = tf.linalg.qr(tf.matmul(J, Q))
        else:
            Q_new = Q
            R_new = tf.zeros_like(Q)
        Rdiag=tf.linalg.diag_part(R_new)
        
        iteration += 1
        return Rdiag, [Q_new, iteration]



















"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
class QRDcell(Layer):
# input at each timepoint should be of the size (batch_size, nUnit, nUnit)
    def __init__(self,start=0, **kwargs):
        super(QRDcell, self).__init__(**kwargs)
        self.start=start


        
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # shape of inputs: (batch_size, nUnit, nUnit)
        self.nUnit=tf.shape(inputs)[-1]
        iteration = tf.zeros((batch_size, 1), dtype=tf.int32)
        return [tf.linalg.eye(self.nUnit, self.nUnit, batch_shape=[batch_size]),iteration]


    @property
    def state_size(self):
        return [tf.TensorShape([self.nUnit, self.nUnit]), tf.TensorShape([None,1])]
    
    def call(self, inputs, states):
        # inputs is  J_{n} (batch, nUnit, nUnit)
        # states is Q_{n-1} (batch, nUnit,nUnit). Q_{0} is identity matrix
        # Q_{n}R_{n} = J_{n}Q_{n-1}
        # Q_{n} is the next state. (Q_new)
        # R_{n} is the output. (R_new)
          
        J = inputs
        Q = states[0]
        iteration=states[1]
        
        if iteration[0]>self.start:
            Q_new, R_new = tf.linalg.qr(tf.matmul(J,Q))
        else:
            Q_new=Q
            R_new=tf.zeros_like(Q)
        iteration+=1
        return R_new, [Q_new, iteration]
    """
