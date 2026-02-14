# -*- coding: utf-8 -*-
"""
Created on Thu May  2 18:55:54 2024

@author: Hiroto
"""

import tensorflow as tf
from tensorflow.keras.layers import AbstractRNNCell, Dense, Concatenate, Input
from tensorflow.python.keras import backend, activations, constraints, initializers, regularizers

#from tensorflow.keras.ops import  ops

# Clear all previously registered custom objects
#$tf.keras.saving.get_custom_objects().clear()



#@tf.keras.saving.register_keras_serializable(package="RNNCustom")
class RNNCustom2Fix2_noise_n(AbstractRNNCell):

    def __init__(
        self,
        units,
        RNN_num,
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
        dense_activation="tanh",
        dense_initializer=None,
        dense_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        seed=None,
        tau=10,
        noisesd=0,
        pert_noisesd=0,
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
        self.RNN_num=RNN_num
        self.input_activation = activations.get(input_activation)
        self.output_activation = activations.get(output_activation)
        self.use_bias = use_bias
        self.tau=tau
        self.noisesd=noisesd
        self.kernel_trainable=kernel_trainable
        self.seed=seed
        self.pert_noisesd=pert_noisesd

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        
        self.dense_activation=dense_activation
        self.dense_initializer=dense_initializer
        self.dense_constraint=dense_constraint
        
        

        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
#        self.state_size = self.units
#        self.output_size = self.units
    
    
    @property
    def state_size(self):
        return [self.units*self.RNN_num]
    
    
    def build(self, input_shape):
        # first dimension of input shape matters, 2nd to last dimension specifies index for perturbation
        self.kernel = self.add_weight(
            shape=(1, self.units*self.RNN_num),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=self.kernel_trainable
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units*self.RNN_num, self.units*self.RNN_num),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units*self.RNN_num,),
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True
    

        self.dense_layers = [
            Dense(2, activation=self.dense_activation, kernel_initializer=self.dense_initializer, kernel_constraint=self.dense_constraint)
            for _ in range(self.RNN_num)
        ]

    def call(self, inputs, states, training=False):
        prev_state = states[0]

        if self.output_activation is not None:
            act_state = self.output_activation(prev_state)

        # split inputs
        input_0 = inputs[:, 0:1]  # Shape: (batch_size, 1), this is the real input
        # get perturbation indices
        input_ind= tf.expand_dims(inputs[:,1:],axis=1) # shape: (batch_size,1, RNN_num)

        h = backend.dot(input_0, self.kernel)# get input kernel
        W=self.recurrent_kernel # get recurrent kernel
        
        if self.bias is not None:
            h += self.bias

        output = h+backend.dot(act_state,W)+tf.random.normal([tf.shape(prev_state)[0],self.units*self.RNN_num],mean=0,stddev=self.noisesd)
 
        if self.input_activation is not None:
            output = self.input_activation(output)
        
        output = prev_state+(output-prev_state)/self.tau
        
        # create noise tensor
        noise=input_ind*tf.random.normal([tf.shape(prev_state)[0],self.units,self.RNN_num],mean=0,stddev=self.pert_noisesd)
        #shape (batch, nUnit, RNN_num)
        # reshape noise to (batch, uNit* RNN_num)
        noise=tf.transpose(tf.reshape(tf.transpose(noise,perm=[2,1,0]), (self.units*self.RNN_num,tf.shape(prev_state)[0])))
        # add noise to states
        output+=noise


        
        # split the output and pass it to dense layer
        split_outputs = tf.split(act_state, num_or_size_splits=self.RNN_num, axis=-1)
        dense_outputs = [dense_layer(split) for dense_layer, split in zip(self.dense_layers, split_outputs)]
        concatenated_output = Concatenate()(dense_outputs)
        
        # return output and state to use for next iteration (name is confusing but act_state is the output, and output is the state to pass)
        return [concatenated_output], [output] # put real output and then states you want to pass to next iteration


    

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "RNN_num": self.RNN_num,
            "input_activation": activations.serialize(self.input_activation),
            "output_activation": activations.serialize(self.output_activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "recurrent_initializer": initializers.serialize(self.recurrent_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "recurrent_regularizer": regularizers.serialize(self.recurrent_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(self.recurrent_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "tau": self.tau,
            "noisesd": self.noisesd,
            "dense_activation": activations.serialize(self.dense_activation),
            "dense_initializer": initializers.serialize(self.dense_initializer),
            "dense_constraint": constraints.serialize(self.dense_constraint),
        })
        return config
    
    
    




#@tf.keras.saving.register_keras_serializable(package="RNNCustom")
class RNNCustom2Fix2_noise_n_getstate(AbstractRNNCell):

    def __init__(
        self,
        units,
        RNN_num,
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
        dense_activation="tanh",
        dense_initializer=None,
        dense_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        seed=None,
        tau=10,
        noisesd=0,
        pert_noisesd=0,
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
        self.RNN_num=RNN_num
        self.input_activation = activations.get(input_activation)
        self.output_activation = activations.get(output_activation)
        self.use_bias = use_bias
        self.tau=tau
        self.noisesd=noisesd
        self.kernel_trainable=kernel_trainable
        self.seed=seed
        self.pert_noisesd=pert_noisesd

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        
        self.dense_activation=dense_activation
        self.dense_initializer=dense_initializer
        self.dense_constraint=dense_constraint
        
        

        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
#        self.state_size = self.units
#        self.output_size = self.units
    
    
    @property
    def state_size(self):
        return [self.units*self.RNN_num]
    
    
    def build(self, input_shape):
        # first dimension of input shape matters, 2nd to last dimension specifies index for perturbation
        self.kernel = self.add_weight(
            shape=(1, self.units*self.RNN_num),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=self.kernel_trainable
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units*self.RNN_num, self.units*self.RNN_num),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units*self.RNN_num,),
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True
    

        self.dense_layers = [
            Dense(2, activation=self.dense_activation, kernel_initializer=self.dense_initializer, kernel_constraint=self.dense_constraint)
            for _ in range(self.RNN_num)
        ]

    def call(self, inputs, states, training=False):
        prev_state = states[0]

        if self.output_activation is not None:
            act_state = self.output_activation(prev_state)

        # split inputs
        input_0 = inputs[:, 0:1]  # Shape: (batch_size, 1), this is the real input
        # get perturbation indices
        input_ind= tf.expand_dims(inputs[:,1:],axis=1) # shape: (batch_size,1, RNN_num)

        h = backend.dot(input_0, self.kernel)# get input kernel
        W=self.recurrent_kernel # get recurrent kernel
        
        if self.bias is not None:
            h += self.bias

        output = h+backend.dot(act_state,W)+tf.random.normal([tf.shape(prev_state)[0],self.units*self.RNN_num],mean=0,stddev=self.noisesd)
 
        if self.input_activation is not None:
            output = self.input_activation(output)
        
        output = prev_state+(output-prev_state)/self.tau
        
        # create noise tensor
        noise=input_ind*tf.random.normal([tf.shape(prev_state)[0],self.units,self.RNN_num],mean=0,stddev=self.pert_noisesd)
        #shape (batch, nUnit, RNN_num)
        # reshape noise to (batch, uNit* RNN_num)
        noise=tf.transpose(tf.reshape(tf.transpose(noise,perm=[2,1,0]), (self.units*self.RNN_num,tf.shape(prev_state)[0])))
        # add noise to states
        output+=noise


        
        # split the output and pass it to dense layer
        split_outputs = tf.split(act_state, num_or_size_splits=self.RNN_num, axis=-1)
        dense_outputs = [dense_layer(split) for dense_layer, split in zip(self.dense_layers, split_outputs)]
        concatenated_output = Concatenate()(dense_outputs)
        
        # return output and state to use for next iteration (name is confusing but act_state is the output, and output is the state to pass)
        return [concatenated_output, output], [output] # put real output and then states you want to pass to next iteration


    

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "RNN_num": self.RNN_num,
            "input_activation": activations.serialize(self.input_activation),
            "output_activation": activations.serialize(self.output_activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "recurrent_initializer": initializers.serialize(self.recurrent_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "recurrent_regularizer": regularizers.serialize(self.recurrent_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(self.recurrent_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "tau": self.tau,
            "noisesd": self.noisesd,
            "dense_activation": activations.serialize(self.dense_activation),
            "dense_initializer": initializers.serialize(self.dense_initializer),
            "dense_constraint": constraints.serialize(self.dense_constraint),
        })
        return config