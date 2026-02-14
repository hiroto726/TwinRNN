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



#@tf.keras.saving.register_keras_serializable(package="RNNCustom")
class RNNCustom2FixPerturb(AbstractRNNCell):

    def __init__(
        self,
        units,
        input_activation="tanh",
        output_activation="tanh",
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
        kernel_trainable=True, # specify whether to train input kernel or not
        dropout=0.0,
        recurrent_dropout=0.0,
        seed=None,
        tau=10,
        noisesd=0,
        perturb_ind=[0, 0],
        pert_state=0,
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
        self.input_activation = activations.get(input_activation)
        self.output_activation = activations.get(output_activation)
        self.use_bias = use_bias
        self.tau=tau
        self.noisesd=noisesd
        self.kernel_trainable=kernel_trainable
        self.seed=seed
        self.pertind=tf.convert_to_tensor(perturb_ind,dtype=tf.int32)
        self.pert_state=pert_state

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        
        

        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
#        self.state_size = self.units
#        self.output_size = self.units
    
    
    @property
    def state_size(self):
        return [self.units,self.units,1,self.units]
    
    
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
        po_A_raw = states[0]
        po_B_raw = states[1]
        iteration=states[2]
        save_state=states[3]
        perturb_ind=self.pertind

        po_A_raw = states[0]
        po_B_raw = states[1]

        if self.output_activation is not None:
            po_A = self.output_activation(po_A_raw)
            po_B = self.output_activation(po_B_raw) 

        h = backend.dot(inputs, self.kernel)
        h_A,h_B=tf.split(h, num_or_size_splits=2, axis=1)
        
        Wr_A, Wr_B, S_A, S_B=tf.split(self.recurrent_kernel,num_or_size_splits=4, axis=1)
        
       
        if self.bias is not None:
            bias_A, bias_B=tf.split(self.bias,num_or_size_splits=2, axis=0)
            h_A += bias_A
            h_B += bias_B

        output_A = h_A + backend.dot(po_A, Wr_A)+backend.dot(po_B, S_B)+tf.random.normal([1,self.units],mean=0,stddev=self.noisesd)
        output_B = h_B + backend.dot(po_B, Wr_B)+backend.dot(po_A, S_A)+tf.random.normal([1,self.units],mean=0,stddev=self.noisesd)
        
        
        if self.input_activation is not None:
            output_A = self.input_activation(output_A)
            output_B = self.input_activation(output_B)
        
        output_A=po_A_raw+(output_A-po_A_raw)/self.tau
        output_B=po_B_raw+(output_B-po_B_raw)/self.tau
        
        iteration+=1
        if self.pert_state==0:
            save_state=tf.where(tf.math.equal(iteration,perturb_ind[:,0:1]),output_A,save_state)
            output_A=tf.where(tf.math.equal(iteration,perturb_ind[:,1:2]),save_state,output_A)
        else:
            save_state=tf.where(tf.math.equal(iteration,perturb_ind[:,0:1]),output_B,save_state)
            output_B=tf.where(tf.math.equal(iteration,perturb_ind[:,1:2]),save_state,output_B)            
        
        return [po_A,po_B], [output_A,output_B,iteration,save_state] # put real output and then states you want to pass to next iteration


    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        po_A_raw = tf.zeros((batch_size, self.units), dtype=dtype)
        po_B_raw = tf.zeros((batch_size, self.units), dtype=dtype)
        iteration = tf.zeros((batch_size, 1), dtype=tf.int32)
        save_state = tf.zeros((batch_size, self.units), dtype=dtype)
        return [po_A_raw, po_B_raw, iteration, save_state]

    
    def get_config(self):
        config = super().get_config()        
        config.update({
            "units": self.units,
            "input_activation": activations.serialize(self.input_activation),
            "output_activation": activations.serialize(self.output_activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "recurrent_initializer": initializers.serialize(
                self.recurrent_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "recurrent_regularizer": regularizers.serialize(
                self.recurrent_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(
                self.recurrent_constraint
            ),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "tau": self.tau,
            "noisesd":self.noisesd
        })
        return config    