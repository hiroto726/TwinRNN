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
class RNNCustom2Fix2_jacobian(AbstractRNNCell):

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
        alpha=0.2,
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
        self.alpha=alpha

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
        
        
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """
        Creates initial states:
          po_A = 0
          po_B = 0
        """
        if batch_size is None:
            # If we can't infer batch_size, you might set some default
            # or raise an error. Here we do a fallback to 1 for demonstration.
            batch_size = 1

        if dtype is None:
            # A fallback if you prefer float32, for example
            dtype = tf.float32

        po_A_init = tf.zeros((batch_size, self.units), dtype=dtype)
        po_B_init = tf.zeros((batch_size, self.units), dtype=dtype)
        return [po_A_init, po_B_init]
        

    def call(self, inputs, states, training=False):
        po_A_raw = states[0] #(batch_num,nUnit)
        po_B_raw = states[1] #(batch_num,nUnit)
        po_raw=tf.concat([po_A_raw,po_B_raw],1) # (batch_num,2*nUnits)

        h = backend.dot(inputs, self.kernel) #: inputs:(batch_num,input_dim) kernel:(input_dim, 2*nUnit)-> h: (batch_num, 2*nUnits)        
    
        Wr_A, Wr_B, S_A, S_B=tf.split(self.recurrent_kernel,num_or_size_splits=4, axis=1)
        # concatenate them to make a single big matrix
        row1=tf.concat([Wr_A,S_A], axis=1)
        row2=tf.concat([S_B,Wr_B], axis=1)
        Wr=tf.concat([row1,row2], axis=0) #(2*nUnits,2*nUnits)
        
        if self.bias is not None:
            h+=self.bias #batch_num,2*nUnits
        
        
        # get jacobian matrix ()
        with tf.GradientTape() as tape:
            tape.watch(po_raw)
            if self.output_activation is not None:
                po = self.output_activation(po_raw) # (batch_num,2*nUnits)        
    
            output= h + backend.dot(po, Wr)+ tf.random.normal([1,2*self.units],mean=0,stddev=self.noisesd) #shape: (batch_num, 2*nUnits)       
            
            if self.input_activation is not None:
                #output_A = self.input_activation(output_A)# there is bug in tensorflow that prevents calculation of jacobian for leaky relu
                output=tf.where(output>0,output,self.alpha*output)
            output=po_raw+(output-po_raw)/self.tau     # (batch_num,2*nUnits)
            
        Jacobian=tape.batch_jacobian(output, po_raw)
        Jacobian = tf.stop_gradient(Jacobian)
        #derivative of output with respect to po_raw, (batch, 2*unit, 2*unit)=(batch_num, input_dim, output_dim)
        
        
        
        po_list=tf.split(po, num_or_size_splits=2, axis=1)# 2*(batch_num,nUnits)
        output_list=tf.split(output, num_or_size_splits=2, axis=1)# 2*(batch_num,nUnits)
        
        return [po_list[0],po_list[1],Jacobian], output_list # put real output and then states you want to pass to next iteration


    
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
    
    
    
import tensorflow as tf
from tensorflow.keras.layers import AbstractRNNCell
from tensorflow.python.keras import backend, activations, constraints, initializers, regularizers

#from tensorflow.keras.ops import  ops

# Clear all previously registered custom objects
#$tf.keras.saving.get_custom_objects().clear()



#@tf.keras.saving.register_keras_serializable(package="RNNCustom")
class RNNCustom2Fix2_jacobian_flat(AbstractRNNCell):

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
        alpha=0.2,
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
        self.alpha=alpha

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
    
    @property
    def output_size(self):
        # Output: [poA, poB, Jacobian_flat]
        return [self.units, self.units, 4 * self.units * self.units]   
    
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
        
        
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """
        Creates initial states:
          po_A = 0
          po_B = 0
        """
        if batch_size is None:
            # If we can't infer batch_size, you might set some default
            # or raise an error. Here we do a fallback to 1 for demonstration.
            batch_size = 1

        if dtype is None:
            # A fallback if you prefer float32, for example
            dtype = tf.float32

        po_A_init = tf.zeros((batch_size, self.units), dtype=dtype)
        po_B_init = tf.zeros((batch_size, self.units), dtype=dtype)
        return [po_A_init, po_B_init]
        

    def call(self, inputs, states, training=False):
        po_A_raw = states[0] #(batch_num,nUnit)
        po_B_raw = states[1] #(batch_num,nUnit)
        po_raw=tf.concat([po_A_raw,po_B_raw],1) # (batch_num,2*nUnits)

        h = backend.dot(inputs, self.kernel) #: inputs:(batch_num,input_dim) kernel:(input_dim, 2*nUnit)-> h: (batch_num, 2*nUnits)        
    
        Wr_A, Wr_B, S_A, S_B=tf.split(self.recurrent_kernel,num_or_size_splits=4, axis=1)
        # concatenate them to make a single big matrix
        row1=tf.concat([Wr_A,S_A], axis=1)
        row2=tf.concat([S_B,Wr_B], axis=1)
        Wr=tf.concat([row1,row2], axis=0) #(2*nUnits,2*nUnits)
        
        if self.bias is not None:
            h+=self.bias #batch_num,2*nUnits
        
        
        # get jacobian matrix ()
        with tf.GradientTape() as tape:
            tape.watch(po_raw)
            if self.output_activation is not None:
                po = self.output_activation(po_raw) # (batch_num,2*nUnits)        
    
            output= h + backend.dot(po, Wr)+ tf.random.normal([1,2*self.units],mean=0,stddev=self.noisesd) #shape: (batch_num, 2*nUnits)       
            
            if self.input_activation is not None:
                #output_A = self.input_activation(output_A)# there is bug in tensorflow that prevents calculation of jacobian for leaky relu
                output=tf.where(output>0,output,self.alpha*output)
            output=po_raw+(output-po_raw)/self.tau     # (batch_num,2*nUnits)
            
        Jacobian=tape.batch_jacobian(output, po_raw)
        Jacobian = tf.stop_gradient(Jacobian)
        #derivative of output with respect to po_raw, (batch, 2*unit, 2*unit)=(batch_num, output_dim, input_dim)
        # Flatten Jacobian for output
        Jacobian_flat = tf.reshape(Jacobian, [tf.shape(Jacobian)[0], -1])  # (batch, 4*units^2)
        #tf.print("Jacobian l2norm:", tf.math.reduce_sum(tf.math.square(Jacobian_flat)))

        
        
        po_list=tf.split(po, num_or_size_splits=2, axis=1)# 2*(batch_num,nUnits)
        output_list=tf.split(output, num_or_size_splits=2, axis=1)# 2*(batch_num,nUnits)
        
        return [po_list[0],po_list[1],Jacobian_flat], output_list # put real output and then states you want to pass to next iteration


    
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
    

import tensorflow as tf
from tensorflow.keras.layers import AbstractRNNCell
from tensorflow.python.keras import backend, activations, constraints, initializers, regularizers

#from tensorflow.keras.ops import  ops

# Clear all previously registered custom objects
#$tf.keras.saving.get_custom_objects().clear()



#@tf.keras.saving.register_keras_serializable(package="RNNCustom")
class RNNCustom2Fix2_jacobian_flat_1out(AbstractRNNCell):

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
        alpha=0.2,
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
        self.alpha=alpha

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
    
    @property
    def output_size(self):
        # Output: [poA, poB, Jacobian_flat]
        return 4 * self.units * self.units  
    
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
        
        
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """
        Creates initial states:
          po_A = 0
          po_B = 0
        """
        if batch_size is None:
            # If we can't infer batch_size, you might set some default
            # or raise an error. Here we do a fallback to 1 for demonstration.
            batch_size = 1

        if dtype is None:
            # A fallback if you prefer float32, for example
            dtype = tf.float32

        po_A_init = tf.zeros((batch_size, self.units), dtype=dtype)
        po_B_init = tf.zeros((batch_size, self.units), dtype=dtype)
        return [po_A_init, po_B_init]
        

    def call(self, inputs, states, training=False):
        po_A_raw = states[0] #(batch_num,nUnit)
        po_B_raw = states[1] #(batch_num,nUnit)
        po_raw=tf.concat([po_A_raw,po_B_raw],1) # (batch_num,2*nUnits)

        h = backend.dot(inputs, self.kernel) #: inputs:(batch_num,input_dim) kernel:(input_dim, 2*nUnit)-> h: (batch_num, 2*nUnits)        
    
        Wr_A, Wr_B, S_A, S_B=tf.split(self.recurrent_kernel,num_or_size_splits=4, axis=1)
        # concatenate them to make a single big matrix
        row1=tf.concat([Wr_A,S_A], axis=1)
        row2=tf.concat([S_B,Wr_B], axis=1)
        Wr=tf.concat([row1,row2], axis=0) #(2*nUnits,2*nUnits)
        
        if self.bias is not None:
            h+=self.bias #batch_num,2*nUnits
        
        
        # get jacobian matrix ()
        with tf.GradientTape() as tape:
            tape.watch(po_raw)
            if self.output_activation is not None:
                po = self.output_activation(po_raw) # (batch_num,2*nUnits)        
    
            output= h + backend.dot(po, Wr)+ tf.random.normal([1,2*self.units],mean=0,stddev=self.noisesd) #shape: (batch_num, 2*nUnits)       
            
            if self.input_activation is not None:
                #output_A = self.input_activation(output_A)# there is bug in tensorflow that prevents calculation of jacobian for leaky relu
                output=tf.where(output>0,output,self.alpha*output)
            output=po_raw+(output-po_raw)/self.tau     # (batch_num,2*nUnits)
            
        Jacobian=tape.batch_jacobian(output, po_raw)
        Jacobian = tf.stop_gradient(Jacobian)
        #derivative of output with respect to po_raw, (batch, 2*unit, 2*unit)=(batch_num, input_dim, output_dim)
        # Flatten Jacobian for output
        Jacobian_flat = tf.reshape(Jacobian, [tf.shape(Jacobian)[0], -1])  # (batch, 4*units^2)

        
        
        po_list=tf.split(po, num_or_size_splits=2, axis=1)# 2*(batch_num,nUnits)
        output_list=tf.split(output, num_or_size_splits=2, axis=1)# 2*(batch_num,nUnits)
        
        return Jacobian_flat, output_list # put real output and then states you want to pass to next iteration


    
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