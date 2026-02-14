# -*- coding: utf-8 -*-
"""
Created on Thu May  2 18:55:54 2024

@author: Fumiya
"""

import tensorflow as tf
from tensorflow.keras.layers import AbstractRNNCell
from tensorflow.python.keras import backend, activations, constraints, initializers, regularizers

#from tensorflow.keras.ops import  ops

# Clear all previously registered custom objects
#$tf.keras.saving.get_custom_objects().clear()


# this rnn have two rnns in one and two dense layers for outputs.
#@tf.keras.saving.register_keras_serializable(package="RNNCustom")
class RNNCustom3(AbstractRNNCell):

    def __init__(
        self,
        units,
        activation="tanh",
        dense_activation="tanh",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        dense_initializer="glorot_uniform",
        dense_bias_initializer="zeros",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        dense_regularizer=None,
        dense_bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dense_constraint=None,
        dense_bias_constraint=None,
        kernel_trainable=True, # specify whether to train input kernel or not
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
        self.dense_activation=activations.get(dense_activation)
        self.use_bias = use_bias
        self.tau=tau
        self.noisesd=noisesd
        self.kernel_trainable=kernel_trainable
        self.seed=seed

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.dense_initializer=initializers.get(dense_initializer)
        self.dense_bias_initializer=initializers.get(dense_bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.dense_regularizer =regularizers.get(dense_regularizer)
        self.dense_bias_regularizer=regularizers.get(dense_bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.dense_constraint= constraints.get(dense_constraint)
        self.dense_bias_constraint=constraints.get(dense_bias_constraint)
        
        

        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
#        self.state_size = self.units
#        self.output_size = self.units
    
    
    @property
    def state_size(self):
        return [self.units,self.units]
    
    
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units*2),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=self.kernel_trainable
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units*4),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        self.dense_kernel = self.add_weight(
            shape=(self.units,4),
            name="dense_kernel",
            initializer=self.dense_initializer,
            regularizer=self.dense_regularizer,
            constraint=self.dense_constraint,
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
            
        self.dense_bias = self.add_weight(
            shape=(4,),
            name="dense_bias",
            initializer=self.dense_bias_initializer,
            regularizer=self.dense_bias_regularizer,
            constraint=self.dense_bias_constraint,
        )
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
        
        newstate_A=po_A+(output_A-po_A)/self.tau
        newstate_B=po_B+(output_B-po_B)/self.tau
        
        dense_kernel_A, dense_kernel_B=tf.split(self.dense_kernel,num_or_size_splits=2, axis=1)
        dense_bias_A, dense_bias_B=tf.split(self.dense_bias,num_or_size_splits=2, axis=0)
        output_A=backend.dot(newstate_A,dense_kernel_A)+dense_bias_A # the addition aligns to the last dimension
        output_B=backend.dot(newstate_B,dense_kernel_B)+dense_bias_B
        
        if self.dense_activation is not None:
            output_A=self.dense_activation(output_A)
            output_B=self.dense_activation(output_B)
            
        output=tf.concat([output_A, output_B],1)
        
        
        return output, [newstate_A, newstate_B]


    
    def get_config(self):
        config = super().get_config()        
        config.update({
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "recurrent_initializer": initializers.serialize(
                self.recurrent_initializer
            ),
            "dense_initializer": initializers.serialize(
                self.dense_initializer
            ),
            "dense_bias_initializer": initializers.serialize(
                self.dense_bias_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "recurrent_regularizer": regularizers.serialize(
                self.recurrent_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "dense_regularizer": regularizers.serialize(
                self.dense_regularizer
            ),
            "dense_bias_regularizer": regularizers.serialize(
                self.dense_bias_regularizer
            ),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(
                self.recurrent_constraint
            ),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dense_constraint": constraints.serialize(
                self.dense_constraint
            ),
            "dense_bias_constraint": constraints.serialize(
                self.dense_bias_constraint
            ),
            "tau": self.tau,
            "noisesd":self.noisesd
        })
        return config    