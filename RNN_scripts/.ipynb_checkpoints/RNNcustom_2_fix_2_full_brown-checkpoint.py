# -*- coding: utf-8 -*-
"""
Created on Thu May  2 18:55:54 2024

@author: Fumiya
"""

import tensorflow as tf
from tensorflow.keras.layers import AbstractRNNCell
from tensorflow.python.keras import backend, activations, constraints, initializers, regularizers

# for stateflu rnn
#from tensorflow.keras.ops import  ops

# Clear all previously registered custom objects
#$tf.keras.saving.get_custom_objects().clear()



#@tf.keras.saving.register_keras_serializable(package="RNNCustom")
class RNNCustom2Fix2full_brown(AbstractRNNCell):

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
        return [self.units, self.units]

    @property
    def output_size(self):
        return self.units * 2  # Must be an integer
    
    
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(1, self.units*2),
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
        po_A_raw, po_B_raw = states  # Unpack the states
        
        if self.output_activation is not None:
            po_A = self.output_activation(po_A_raw)
            po_B = self.output_activation(po_B_raw)
        
        inputs_0=inputs[:,0:1]# batch_size * input_dim(1)
        inputs_brown=inputs[:,1:]# batch_size * nUnits
        h = backend.dot(inputs_0, self.kernel)
        h_A, h_B = tf.split(h, num_or_size_splits=2, axis=1)
        
        Wr_A, Wr_B, S_A, S_B = tf.split(self.recurrent_kernel, num_or_size_splits=4, axis=1)
        
        if self.bias is not None:
            bias_A, bias_B = tf.split(self.bias, num_or_size_splits=2, axis=0)
            h_A += bias_A
            h_B += bias_B
        
        batch_size = tf.shape(inputs_0)[0]
        
        output_A = h_A + backend.dot(po_A, Wr_A) + backend.dot(po_B, S_B) + tf.random.normal(
            [batch_size, self.units], mean=0, stddev=self.noisesd, seed=self.seed)
        output_B = h_B + backend.dot(po_B, Wr_B) + backend.dot(po_A, S_A) + tf.random.normal(
            [batch_size, self.units], mean=0, stddev=self.noisesd, seed=self.seed)
        
        if self.input_activation is not None:
            output_A = self.input_activation(output_A)
            output_B = self.input_activation(output_B)
        
        output_A = po_A_raw + (output_A - po_A_raw) / self.tau
        output_B = po_B_raw + (output_B - po_B_raw) / self.tau
        
        # add brownian noise
        output_A += inputs_brown
        output_B += inputs_brown
        
        # Concatenate outputs
        outputs = tf.concat([po_A, po_B], axis=-1)  # Shape: (batch_size, units * 2)
        new_states = [output_A, output_B]# Shape: (batch_size, units)
        return outputs, new_states



    
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
    
    
    
    
    
    
#@tf.keras.saving.register_keras_serializable(package="RNNCustom")
class RNNCustom2Fix2full_brown_2(AbstractRNNCell):
# made it so that brown noise is low rank and it has fixed sets of random vectors.

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
        noise_weights=0,# it should be a tensorflow vector of shape (rank, units)
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
        self.noise_weights=tf.cast(noise_weights, dtype=tf.float32) # it should be a tensorflow vector of shape (rank, units)

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
        return [self.units, self.units]

    @property
    def output_size(self):
        return self.units * 2  # Must be an integer
    
    
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(1, self.units*2),
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
        po_A_raw, po_B_raw = states  # Unpack the states
        
        if self.output_activation is not None:
            po_A = self.output_activation(po_A_raw)
            po_B = self.output_activation(po_B_raw)
        
        inputs_0=inputs[:,0:1]# batch_size * input_dim(1)
        inputs_brown=inputs[:,1:]# batch_size * rank
        h = backend.dot(inputs_0, self.kernel)
        h_A, h_B = tf.split(h, num_or_size_splits=2, axis=1)
        
        Wr_A, Wr_B, S_A, S_B = tf.split(self.recurrent_kernel, num_or_size_splits=4, axis=1)
        
        if self.bias is not None:
            bias_A, bias_B = tf.split(self.bias, num_or_size_splits=2, axis=0)
            h_A += bias_A
            h_B += bias_B
        
        batch_size = tf.shape(inputs_0)[0]
        
        output_A = h_A + backend.dot(po_A, Wr_A) + backend.dot(po_B, S_B) + tf.random.normal(
            [batch_size, self.units], mean=0, stddev=self.noisesd, seed=self.seed)
        output_B = h_B + backend.dot(po_B, Wr_B) + backend.dot(po_A, S_A) + tf.random.normal(
            [batch_size, self.units], mean=0, stddev=self.noisesd, seed=self.seed)
        
        if self.input_activation is not None:
            output_A = self.input_activation(output_A)
            output_B = self.input_activation(output_B)
        
        output_A = po_A_raw + (output_A - po_A_raw) / self.tau
        output_B = po_B_raw + (output_B - po_B_raw) / self.tau
        
        # add brownian noise
        brown_noise = backend.dot(inputs_brown,self.noise_weights)#(batch,rank),(rank,units)
        output_A += brown_noise # (batch,units)
        output_B += brown_noise # (batch,units)
        
        # Concatenate outputs
        outputs = tf.concat([po_A, po_B], axis=-1)  # Shape: (batch_size, units * 2)
        new_states = [output_A, output_B]# Shape: (batch_size, units)
        return outputs, new_states



    
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
   
# create uniform norm constraint
from tensorflow.keras.constraints import Constraint 
class UnitNormConstraint(Constraint):
    def __call__(self, w):
        return tf.linalg.l2_normalize(w, axis=1)  # Normalize each row

import numpy as np
class RNNCustom2Fix2full_brown_nonlinear(AbstractRNNCell):
# made it so that brown noise is low rank and it has fixed sets of random vectors.

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
        noise_weights=0,# it should be a tensorflow vector of shape (rank, units)
        noise_ker_A=0,
        noise_ker_B=0,
        noise_ker_norm=10,
        noise_rank=2,
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
        self.noise_weights=tf.cast(noise_weights, dtype=tf.float32) # it should be a tensorflow vector of shape (rank, units)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        
        self.noise_ker_A = np.array(noise_ker_A, dtype=np.float32)
        self.noise_ker_B = np.array(noise_ker_B, dtype=np.float32)
        
        self.noise_ker_norm=noise_ker_norm
        self.noise_rank=int(noise_rank)
        
        

        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
#        self.state_size = self.units
#        self.output_size = self.units
    
    
    @property
    def state_size(self):
        return [self.units, self.units]

    @property
    def output_size(self):
        return self.units * 2  # Must be an integer
    
    
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(1, self.units*2),
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
        self.noise_input_A=self.add_weight(
            shape=(self.noise_rank, self.units),
            name="noise_input_kernel_A",
            initializer=tf.constant_initializer(self.noise_ker_A),
            constraint=UnitNormConstraint(),
            trainable=True)
        self.noise_input_B=self.add_weight(
            shape=(self.noise_rank, self.units),
            name="noise_input_kernel_B",
            initializer=tf.constant_initializer(self.noise_ker_B),
            constraint=UnitNormConstraint(),
            trainable=True)
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
        po_A_raw, po_B_raw = states  # Unpack the states
        
        if self.output_activation is not None:
            po_A = self.output_activation(po_A_raw)
            po_B = self.output_activation(po_B_raw)
        
        
        inputs_0=inputs[:,0:1]# batch_size * input_dim(1)
        inputs_brown_A=inputs[:,1:1+self.noise_rank]# batch_size * rank
        inputs_brown_B=inputs[:,1+self.noise_rank:1+2*self.noise_rank]# batch_size * rank
        
        h = backend.dot(inputs_0, self.kernel)
        h_A, h_B = tf.split(h, num_or_size_splits=2, axis=1)
        
        Wr_A, Wr_B, S_A, S_B = tf.split(self.recurrent_kernel, num_or_size_splits=4, axis=1)
        
        if self.bias is not None:
            bias_A, bias_B = tf.split(self.bias, num_or_size_splits=2, axis=0)
            h_A += bias_A
            h_B += bias_B
        
        batch_size = tf.shape(inputs_0)[0]
        
        output_A = h_A + backend.dot(po_A, Wr_A) + backend.dot(po_B, S_B) + tf.random.normal(
            [batch_size, self.units], mean=0, stddev=self.noisesd, seed=self.seed)
        output_B = h_B + backend.dot(po_B, Wr_B) + backend.dot(po_A, S_A) + tf.random.normal(
            [batch_size, self.units], mean=0, stddev=self.noisesd, seed=self.seed)
        
        if self.input_activation is not None:
            output_A = self.input_activation(output_A)
            output_B = self.input_activation(output_B)
        
        output_A = po_A_raw + (output_A - po_A_raw) / self.tau
        output_B = po_B_raw + (output_B - po_B_raw) / self.tau
        
        # add brownian noise
        brown_noise_A = backend.dot(inputs_brown_A,self.noise_ker_norm*self.noise_input_A)#(batch,rank),(rank,units)
        brown_noise_B = backend.dot(inputs_brown_B,self.noise_ker_norm*self.noise_input_B)#(batch,rank),(rank,units)
        
        
        output_A += brown_noise_A # (batch,units)
        output_B += brown_noise_B # (batch,units)
        
        # Concatenate outputs
        outputs = tf.concat([po_A, po_B], axis=-1)  # Shape: (batch_size, units * 2)
        new_states = [output_A, output_B]# Shape: (batch_size, units)
        return outputs, new_states