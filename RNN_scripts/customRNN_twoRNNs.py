# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:11:26 2024

@author: Fumiya
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# use stateful rnn
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, SimpleRNN, SimpleRNNCell, GaussianNoise, RNN, AbstractRNNCell
from tensorflow.python.keras import backend, activations, constraints, initializers, regularizers
from tensorflow.python.ops import array_ops as ops
#from tensorflow.keras.ops import  ops

class RNNCustom2(AbstractRNNCell):

    def __init__(
        self,
        units,
        activation="tanh",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        seed=None,
        tau=10,
        noisesd=0,
        **kwargs,
    ):
        if units <= 0:
            raise ValueError(
                "Received an invalid value for argument `units`, "
                f"expected a positive integer, got {units}."
            )
        super().__init__(**kwargs)
#        self.seed = seed
#        self.seed_generator = backend.random.SeedGenerator(seed)

        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.tau=tau
        self.noisesd=noisesd

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

#        self.dropout = min(1.0, max(0.0, dropout))
#        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
#        self.state_size = self.units
#        self.output_size = self.units
    
    
    @property
    def state_size(self):
        return self.units
    
    
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units*2),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units*4),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units*2,),
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=False):
        po_A = states[0]
        po_B = states[1]

        h = backend.dot(inputs, self.kernel)
        h_A,h_B=tf.split(h, num_or_size_splits=2, axis=1)
        
        Wr_A, Wr_B, S_A, S_B=tf.split(self.recurrent_kernel,num_or_size_splits=4, axis=1)
        
       
        if self.bias is not None:
            bias_A, bias_B=tf.split(self.bias,num_or_size_splits=2, axis=0)
            h_A += bias_A
            h_B += bias_B

        output_A = h_A + backend.dot(po_A, Wr_A)+backend.dot(po_B, S_B)+tf.random.normal([1,self.units],mean=0,stddev=self.noisesd)
        output_B = h_B + backend.dot(po_B, Wr_B)+backend.dot(po_A, S_A)+tf.random.normal([1,self.units],mean=0,stddev=self.noisesd)
        if self.activation is not None:
            output_A = self.activation(output_A)
            output_B = self.activation(output_B)
        
        output_A=po_A+(output_A-po_A)/self.tau
        output_B=po_B+(output_B-po_B)/self.tau
        return [output_A,output_B], [output_A,output_B]