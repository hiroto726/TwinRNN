# -*- coding: utf-8 -*-
"""
Created on Thu May  2 18:55:54 2024

@author: Hiroto
"""

import tensorflow as tf
from tensorflow.keras.layers import AbstractRNNCell
from tensorflow.python.keras import backend, activations, constraints, initializers, regularizers

#from tensorflow.keras.ops import  ops

# Clear all previously registered custom objects
#$tf.keras.saving.get_custom_objects().clear()



#@tf.keras.saving.register_keras_serializable(package="RNNCustom")
class RNNCustom2sameweights_Jacobian(AbstractRNNCell):

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
        return [self.units,self.units]
    
    
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=self.kernel_trainable
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units*2),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True
        
    #@tf.function
    def call(self, inputs, states, training=False):




        h = backend.dot(inputs, self.kernel)
        
        Wr_A, S_A=tf.split(self.recurrent_kernel,num_or_size_splits=2, axis=1)
        
       
        if self.bias is not None:
            bias=self.bias
            h += bias
            
        po_A_raw = states[0]
        po_B_raw = states[1]
        with tf.GradientTape() as tape:
            tape.watch(po_A_raw)            

            if self.output_activation is not None:
                po_A = self.output_activation(po_A_raw) 
                po_B = self.output_activation(po_B_raw) 

                
            output_A = h + backend.dot(po_A, Wr_A)+backend.dot(po_B, S_A)+tf.random.normal([1,self.units],mean=0,stddev=self.noisesd)
            output_B = h + backend.dot(po_B, Wr_A)+backend.dot(po_A, S_A)+tf.random.normal([1,self.units],mean=0,stddev=self.noisesd)
            
            
            if self.input_activation is not None:
                #output_A = self.input_activation(output_A)# there is bug in tensorflow that prevents calculation of jacobian for leaky relu
                #output_B = self.input_activation(output_B)# there is bug in tensorflow that prevents calculation of jacobian for leaky relu
                output_A=tf.where(output_A>0,output_A,0.2*output_A)
                output_B=tf.where(output_B>0,output_B,0.2*output_B)
            
            output_A=po_A_raw+(output_A-po_A_raw)/self.tau
            output_B=po_B_raw+(output_B-po_B_raw)/self.tau
        jacobian=tape.batch_jacobian(output_A, po_A_raw) 
        #for each time point, the jacobian has the shape (batch, output_dim(=nUnit), input_dim(=nUnit))
       
        
        return [po_A,po_B,jacobian], [output_A,output_B] # put real output and then states you want to pass to next iteration


    
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