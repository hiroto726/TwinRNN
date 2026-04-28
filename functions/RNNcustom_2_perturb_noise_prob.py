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
class RNNCustom2FixPerturb_noise_prob(AbstractRNNCell):

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
        pert_which=[True,True],# boolean to specify which perturbation will perturb RNN A. True pertrub RNN A
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
        self.input_activation = activations.get(input_activation)
        self.output_activation = activations.get(output_activation)
        self.use_bias = use_bias
        self.tau=tau
        self.noisesd=noisesd
        self.kernel_trainable=kernel_trainable
        self.seed=seed
        self.pertind=tf.convert_to_tensor(perturb_ind,dtype=tf.int32)
        self.pert_noisesd=pert_noisesd
        self.pert_A=pert_which
        self.pert_B=tf.math.logical_not(pert_which)

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
        return [self.units,self.units,1]
    
    
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
        boolind=tf.math.equal(iteration,perturb_ind)# perturb_ind is [batch, anynumber]
        
        
        boolind_A=tf.math.reduce_any(tf.math.logical_and(boolind,self.pert_A),axis=1,keepdims=True)
        output_A_noise=output_A+tf.random.normal([1,self.units],mean=0,stddev=self.pert_noisesd)# add noise to RNN A
        #save_state=tf.where(tf.math.equal(iteration,perturb_ind[:,0:1]),output_A,save_state) #output_A [batch_size, nUnit]
        output_A=tf.where(boolind_A,output_A_noise,output_A) # add noise only if iteration=pertrub_ind

        boolind_B=tf.math.reduce_any(tf.math.logical_and(boolind,self.pert_B),axis=1,keepdims=True)
        output_B_noise=output_B+tf.random.normal([1,self.units],mean=0,stddev=self.pert_noisesd)# add noise to RNN B
        #save_state=tf.where(tf.math.equal(iteration,perturb_ind[:,0:1]),output_B,save_state)
        output_B=tf.where(boolind_B,output_B_noise,output_B)            
        
        return [po_A,po_B], [output_A,output_B,iteration] # put real output and then states you want to pass to next iteration


    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        po_A_raw = tf.zeros((batch_size, self.units), dtype=dtype)
        po_B_raw = tf.zeros((batch_size, self.units), dtype=dtype)
        iteration = tf.zeros((batch_size, 1), dtype=tf.int32)
        return [po_A_raw, po_B_raw, iteration]

    
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
class RNNCustom2FixPerturb_noise_prob_unitnoise(AbstractRNNCell):

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
        pert_which=[True,True],# boolean to specify which perturbation will perturb RNN A. True pertrub RNN A
        pert_noisesd=0,
        unit_noise_ind=0, #index of unit to apply noise
        unit_noise_A=0,# 0 to apply unit noise to RNN A and 1 otherwise
        unit_noise_SD=0,# SD of unit noise
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
        self.pert_noisesd=pert_noisesd
        self.pert_A=pert_which
        self.pert_B=tf.math.logical_not(pert_which)
        self.unit_noise_ind=unit_noise_ind
        self.unit_noise_A=unit_noise_A# 0 to apply unit noise to RNN A and 1 otherwise
        self.unit_noise_SD=unit_noise_SD# SD of unit noise

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
        return [self.units,self.units,1]
    
    
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
        perturb_ind=self.pertind

        po_A_raw = states[0]
        po_B_raw = states[1]

        if self.output_activation is not None:
            po_A = self.output_activation(po_A_raw)
            po_B = self.output_activation(po_B_raw) 
            
        batch_num=tf.shape(inputs)[0]
        h = backend.dot(inputs, self.kernel)
        h_A,h_B=tf.split(h, num_or_size_splits=2, axis=1)
        
        Wr_A, Wr_B, S_A, S_B=tf.split(self.recurrent_kernel,num_or_size_splits=4, axis=1)
        
       
        if self.bias is not None:
            bias_A, bias_B=tf.split(self.bias,num_or_size_splits=2, axis=0)
            h_A += bias_A
            h_B += bias_B

        output_A = h_A + backend.dot(po_A, Wr_A)+backend.dot(po_B, S_B)+tf.random.normal([batch_num,self.units],mean=0,stddev=self.noisesd)
        output_B = h_B + backend.dot(po_B, Wr_B)+backend.dot(po_A, S_A)+tf.random.normal([batch_num,self.units],mean=0,stddev=self.noisesd)
        
        # make noise tensor
        noise_column = tf.random.normal([batch_num], mean=0, stddev=self.unit_noise_SD)  # Shape: (batch_num,)
        noise_mask = tf.one_hot(self.unit_noise_ind, depth=self.units, dtype=tf.float32)  # Shape: (self.units,)
        noise_mat = noise_column[:, tf.newaxis] * noise_mask  # Shape: (batch_num, self.units)
        if self.unit_noise_A==0:
            output_A+=noise_mat
        else:
            output_B+=noise_mat
        
        
        if self.input_activation is not None:
            output_A = self.input_activation(output_A)
            output_B = self.input_activation(output_B)
        
        output_A=po_A_raw+(output_A-po_A_raw)/self.tau
        output_B=po_B_raw+(output_B-po_B_raw)/self.tau
        
        
        iteration+=1
        boolind=tf.math.equal(iteration,perturb_ind)# perturb_ind is [batch, anynumber]
        
        
        boolind_A=tf.math.reduce_any(tf.math.logical_and(boolind,self.pert_A),axis=1,keepdims=True)
        output_A_noise=output_A+tf.random.normal([1,self.units],mean=0,stddev=self.pert_noisesd)# add noise to RNN A
        #save_state=tf.where(tf.math.equal(iteration,perturb_ind[:,0:1]),output_A,save_state) #output_A [batch_size, nUnit]
        output_A=tf.where(boolind_A,output_A_noise,output_A) # add noise only if iteration=pertrub_ind

        boolind_B=tf.math.reduce_any(tf.math.logical_and(boolind,self.pert_B),axis=1,keepdims=True)
        output_B_noise=output_B+tf.random.normal([1,self.units],mean=0,stddev=self.pert_noisesd)# add noise to RNN B
        #save_state=tf.where(tf.math.equal(iteration,perturb_ind[:,0:1]),output_B,save_state)
        output_B=tf.where(boolind_B,output_B_noise,output_B)            
        
        return [po_A,po_B], [output_A,output_B,iteration] # put real output and then states you want to pass to next iteration


    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        po_A_raw = tf.zeros((batch_size, self.units), dtype=dtype)
        po_B_raw = tf.zeros((batch_size, self.units), dtype=dtype)
        iteration = tf.zeros((batch_size, 1), dtype=tf.int32)
        return [po_A_raw, po_B_raw, iteration]

    
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
class RNNCustom2FixPerturb_noise_prob_brown(AbstractRNNCell):

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
        pert_which=[True,True],# boolean to specify which perturbation will perturb RNN A. True pertrub RNN A
        pert_noisesd=0,
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
        self.pertind=tf.convert_to_tensor(perturb_ind,dtype=tf.int32)
        self.pert_noisesd=pert_noisesd
        self.pert_A=pert_which
        self.pert_B=tf.math.logical_not(pert_which)
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
        return [self.units,self.units,1]
    
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
        po_A_raw = states[0]
        po_B_raw = states[1]
        iteration=states[2]
        perturb_ind=self.pertind


        if self.output_activation is not None:
            po_A = self.output_activation(po_A_raw)
            po_B = self.output_activation(po_B_raw) 

        inputs_0=inputs[:,0:1]# batch_size * input_dim(1)
        inputs_brown=inputs[:,1:]# batch_size * nUnits

        h = backend.dot(inputs_0, self.kernel)
        h_A,h_B=tf.split(h, num_or_size_splits=2, axis=1)
        
        Wr_A, Wr_B, S_A, S_B=tf.split(self.recurrent_kernel,num_or_size_splits=4, axis=1)
        
       
        if self.bias is not None:
            bias_A, bias_B=tf.split(self.bias,num_or_size_splits=2, axis=0)
            h_A += bias_A
            h_B += bias_B

        batch_size = tf.shape(inputs_0)[0]
        
        output_A = h_A + backend.dot(po_A, Wr_A)+backend.dot(po_B, S_B)+tf.random.normal([batch_size,self.units],mean=0,stddev=self.noisesd, seed=self.seed)
        output_B = h_B + backend.dot(po_B, Wr_B)+backend.dot(po_A, S_A)+tf.random.normal([batch_size,self.units],mean=0,stddev=self.noisesd, seed=self.seed)
        
        
        if self.input_activation is not None:
            output_A = self.input_activation(output_A)
            output_B = self.input_activation(output_B)
        
        output_A=po_A_raw+(output_A-po_A_raw)/self.tau
        output_B=po_B_raw+(output_B-po_B_raw)/self.tau
        
        
        iteration+=1
        boolind=tf.math.equal(iteration,perturb_ind)# perturb_ind is [batch, anynumber]
        
        
        boolind_A=tf.math.reduce_any(tf.math.logical_and(boolind,self.pert_A),axis=1,keepdims=True)
        output_A_noise=output_A+tf.random.normal([batch_size,self.units],mean=0,stddev=self.pert_noisesd)# add noise to RNN A
        #save_state=tf.where(tf.math.equal(iteration,perturb_ind[:,0:1]),output_A,save_state) #output_A [batch_size, nUnit]
        output_A=tf.where(boolind_A,output_A_noise,output_A) # add noise only if iteration=pertrub_ind

        boolind_B=tf.math.reduce_any(tf.math.logical_and(boolind,self.pert_B),axis=1,keepdims=True)
        output_B_noise=output_B+tf.random.normal([batch_size,self.units],mean=0,stddev=self.pert_noisesd)# add noise to RNN B
        #save_state=tf.where(tf.math.equal(iteration,perturb_ind[:,0:1]),output_B,save_state)
        output_B=tf.where(boolind_B,output_B_noise,output_B)

        # add brownian noise
        brown_noise = backend.dot(inputs_brown,self.noise_weights)#(batch,rank),(rank,units)
        output_A += brown_noise # (batch,units)
        output_B += brown_noise # (batch,units)           
        
        outputs = tf.concat([po_A, po_B], axis=-1)# Shape: (batch_size, units * 2) 
        return outputs, [output_A,output_B,iteration] # put real output and then states you want to pass to next iteration


    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        po_A_raw = tf.zeros((batch_size, self.units), dtype=dtype)
        po_B_raw = tf.zeros((batch_size, self.units), dtype=dtype)
        iteration = tf.zeros((batch_size, 1), dtype=tf.int32)
        return [po_A_raw, po_B_raw, iteration]

    
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




class RNNCustom2FixPerturb_noise_dir_prob_brown(AbstractRNNCell):

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
        pert_which=[True,True],# boolean to specify which perturbation will perturb RNN A. True pertrub RNN A
        pert_noisesd=0,
        noise_weights=0,# it should be a tensorflow vector of shape (rank, units)
        noise_vec=None, # direction to embed the noise size=(1,nUnit)
        sync_noise=True,# sync the noise between RNN A and B or not
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
        self.pert_noisesd=pert_noisesd
        self.pert_A=pert_which
        self.pert_B=tf.math.logical_not(pert_which)
        self.noise_weights=tf.cast(noise_weights, dtype=tf.float32) # it should be a tensorflow vector of shape (rank, units)
        self.sync_noise=sync_noise

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        if noise_vec is None:
            self.noise_vec_A=tf.zeros((1,units),dtype=tf.float32)
            self.noise_vec_B=tf.zeros((1,units),dtype=tf.float32)
        else:
            self.noise_vec_A=tf.convert_to_tensor(noise_vec[0],dtype=tf.float32)
            self.noise_vec_B=tf.convert_to_tensor(noise_vec[1], dtype=tf.float32)
        
        
        

        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
#        self.state_size = self.units
#        self.output_size = self.units
    
    
    @property
    def state_size(self):
        return [self.units,self.units,1]
    
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
        po_A_raw = states[0]
        po_B_raw = states[1]
        iteration=states[2]
        perturb_ind=self.pertind


        if self.output_activation is not None:
            po_A = self.output_activation(po_A_raw)
            po_B = self.output_activation(po_B_raw) 

        inputs_0=inputs[:,0:1]# batch_size * input_dim(1)
        inputs_brown=inputs[:,1:]# batch_size * nUnits

        h = backend.dot(inputs_0, self.kernel)
        h_A,h_B=tf.split(h, num_or_size_splits=2, axis=1)
        
        Wr_A, Wr_B, S_A, S_B=tf.split(self.recurrent_kernel,num_or_size_splits=4, axis=1)
        
       
        if self.bias is not None:
            bias_A, bias_B=tf.split(self.bias,num_or_size_splits=2, axis=0)
            h_A += bias_A
            h_B += bias_B

        batch_size = tf.shape(inputs_0)[0]
        
        output_A = h_A + backend.dot(po_A, Wr_A)+backend.dot(po_B, S_B)+tf.random.normal([batch_size,self.units],mean=0,stddev=self.noisesd, seed=self.seed)
        output_B = h_B + backend.dot(po_B, Wr_B)+backend.dot(po_A, S_A)+tf.random.normal([batch_size,self.units],mean=0,stddev=self.noisesd, seed=self.seed)
        
        
        if self.input_activation is not None:
            output_A = self.input_activation(output_A)
            output_B = self.input_activation(output_B)
        
        output_A=po_A_raw+(output_A-po_A_raw)/self.tau
        output_B=po_B_raw+(output_B-po_B_raw)/self.tau
        
        
        iteration+=1
        boolind=tf.math.equal(iteration,perturb_ind)# perturb_ind is [batch, anynumber]
        
        
        boolind_A=tf.math.reduce_any(tf.math.logical_and(boolind,self.pert_A),axis=1,keepdims=True)
        boolind_B=tf.math.reduce_any(tf.math.logical_and(boolind,self.pert_B),axis=1,keepdims=True)
        
        noise_scale_A=tf.random.normal([batch_size,1],mean=0,stddev=self.pert_noisesd)
        noise_scale_B=tf.random.normal([batch_size,1],mean=0,stddev=self.pert_noisesd)
        if self.sync_noise is True:
            boolind_both=tf.math.logical_or(boolind_A, boolind_B)
            boolind_A=boolind_both
            boolind_B=boolind_both
            noise_scale_B=noise_scale_A
        
        noise_add_A=backend.dot(noise_scale_A,self.noise_vec_A)
        output_A_noise=output_A+noise_add_A
        output_A=tf.where(boolind_A,output_A_noise,output_A) # add noise only if iteration=pertrub_ind


        
        noise_add_B=backend.dot(noise_scale_B,self.noise_vec_B)
        output_B_noise=output_B+noise_add_B
        output_B=tf.where(boolind_B,output_B_noise,output_B)


        # add brownian noise
        brown_noise = backend.dot(inputs_brown,self.noise_weights)#(batch,rank),(rank,units)
        output_A += brown_noise # (batch,units)
        output_B += brown_noise # (batch,units)           
        
        outputs = tf.concat([po_A, po_B], axis=-1)# Shape: (batch_size, units * 2) 
        return outputs, [output_A,output_B,iteration] # put real output and then states you want to pass to next iteration


    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        po_A_raw = tf.zeros((batch_size, self.units), dtype=dtype)
        po_B_raw = tf.zeros((batch_size, self.units), dtype=dtype)
        iteration = tf.zeros((batch_size, 1), dtype=tf.int32)
        return [po_A_raw, po_B_raw, iteration]
    
    
    
class RNNCustom2FixPerturb_noise_dir_prob_brown_ratio(AbstractRNNCell):

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
        pert_which=[True,True],# boolean to specify which perturbation will perturb RNN A. True pertrub RNN A
        pert_noisesd=0,
        noise_weights=0,# it should be a tensorflow vector of shape (rank, units)
        pert_noise_mat=None, # direction to embed the noise size=(nUnit,100)
        sync_noise=True,# sync the noise between RNN A and B or not
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
        self.pert_noisesd=pert_noisesd
        self.pert_A=pert_which
        self.pert_B=tf.math.logical_not(pert_which)
        self.noise_weights=tf.cast(noise_weights, dtype=tf.float32) # it should be a tensorflow vector of shape (rank, units)
        self.sync_noise=sync_noise

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        if pert_noise_mat is None:
            self.noise_mat_A=tf.zeros((1,units),dtype=tf.float32)
            self.noise_mat_B=tf.zeros((1,units),dtype=tf.float32)
        else:
            #transpose to (101,nUnit)
            self.noise_mat_A=tf.convert_to_tensor(pert_noise_mat[0].T,dtype=tf.float32)
            self.noise_mat_B=tf.convert_to_tensor(pert_noise_mat[1].T, dtype=tf.float32)
        
        
        

        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
#        self.state_size = self.units
#        self.output_size = self.units
    
    
    @property
    def state_size(self):
        return [self.units,self.units,1]
    
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
        po_A_raw = states[0]
        po_B_raw = states[1]
        iteration=states[2]
        perturb_ind=self.pertind


        if self.output_activation is not None:
            po_A = self.output_activation(po_A_raw)
            po_B = self.output_activation(po_B_raw) 

        inputs_0=inputs[:,0:1]# batch_size * input_dim(1)
        inputs_brown=inputs[:,1:3]# batch_size * rank
        inputs_noise_w=inputs[:,3:]# batch_size, 100

        h = backend.dot(inputs_0, self.kernel)
        h_A,h_B=tf.split(h, num_or_size_splits=2, axis=1)
        
        Wr_A, Wr_B, S_A, S_B=tf.split(self.recurrent_kernel,num_or_size_splits=4, axis=1)
        
       
        if self.bias is not None:
            bias_A, bias_B=tf.split(self.bias,num_or_size_splits=2, axis=0)
            h_A += bias_A
            h_B += bias_B

        batch_size = tf.shape(inputs_0)[0]
        
        output_A = h_A + backend.dot(po_A, Wr_A)+backend.dot(po_B, S_B)+tf.random.normal([batch_size,self.units],mean=0,stddev=self.noisesd, seed=self.seed)
        output_B = h_B + backend.dot(po_B, Wr_B)+backend.dot(po_A, S_A)+tf.random.normal([batch_size,self.units],mean=0,stddev=self.noisesd, seed=self.seed)
        
        
        if self.input_activation is not None:
            output_A = self.input_activation(output_A)
            output_B = self.input_activation(output_B)
        
        output_A=po_A_raw+(output_A-po_A_raw)/self.tau
        output_B=po_B_raw+(output_B-po_B_raw)/self.tau
        
        
        iteration+=1
        boolind=tf.math.equal(iteration,perturb_ind)# perturb_ind is [batch, anynumber]
        
        
        boolind_A=tf.math.reduce_any(tf.math.logical_and(boolind,self.pert_A),axis=1,keepdims=True)
        boolind_B=tf.math.reduce_any(tf.math.logical_and(boolind,self.pert_B),axis=1,keepdims=True)
        
        noise_scale_A=tf.random.normal([batch_size,1],mean=0,stddev=self.pert_noisesd)
        noise_scale_B=tf.random.normal([batch_size,1],mean=0,stddev=self.pert_noisesd)
        if self.sync_noise is True:
            boolind_both=tf.math.logical_or(boolind_A, boolind_B)
            boolind_A=boolind_both
            boolind_B=boolind_both
            noise_scale_B=noise_scale_A
        
        noise_proj_A=tf.matmul(inputs_noise_w, self.noise_mat_A)#(batch, 100)(100,nUnit)->(batch, nUnit)
        noise_add_A=noise_scale_A*noise_proj_A #(batch,1)*(batcn, nUnit)->(batch, nUnit)
        output_A_noise=output_A+noise_add_A
        output_A=tf.where(boolind_A,output_A_noise,output_A) # add noise only if iteration=pertrub_ind


        noise_proj_B=tf.matmul(inputs_noise_w, self.noise_mat_B)#(batch, 100)(100,nUnit)->(batch, nUnit)
        noise_add_B=noise_scale_B*noise_proj_B #(batch,1)*(batcn, nUnit)->(batch, nUnit)        
        output_B_noise=output_B+noise_add_B
        output_B=tf.where(boolind_B,output_B_noise,output_B)


        # add brownian noise
        brown_noise = backend.dot(inputs_brown,self.noise_weights)#(batch,rank),(rank,units)
        output_A += brown_noise # (batch,units)
        output_B += brown_noise # (batch,units)           
        
        outputs = tf.concat([po_A, po_B], axis=-1)# Shape: (batch_size, units * 2) 
        return outputs, [output_A,output_B,iteration] # put real output and then states you want to pass to next iteration


    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        po_A_raw = tf.zeros((batch_size, self.units), dtype=dtype)
        po_B_raw = tf.zeros((batch_size, self.units), dtype=dtype)
        iteration = tf.zeros((batch_size, 1), dtype=tf.int32)
        return [po_A_raw, po_B_raw, iteration]



class RNNCustom2FixPerturb_noise_dir_prob_brown_return_noise(AbstractRNNCell):

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
        pert_which=[True,True],# boolean to specify which perturbation will perturb RNN A. True pertrub RNN A
        pert_noisesd=0,
        noise_weights=0,# it should be a tensorflow vector of shape (rank, units)
        noise_vec=None, # direction to embed the noise size=(1,nUnit)
        sync_noise=True,# sync the noise between RNN A and B or not
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
        self.pert_noisesd=pert_noisesd
        self.pert_A=pert_which
        self.pert_B=tf.math.logical_not(pert_which)
        self.noise_weights=tf.cast(noise_weights, dtype=tf.float32) # it should be a tensorflow vector of shape (rank, units)
        self.sync_noise=sync_noise

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        if noise_vec is None:
            self.noise_vec_A=tf.zeros((1,units),dtype=tf.float32)
            self.noise_vec_B=tf.zeros((1,units),dtype=tf.float32)
        else:
            self.noise_vec_A=tf.convert_to_tensor(noise_vec[0],dtype=tf.float32)
            self.noise_vec_B=tf.convert_to_tensor(noise_vec[1], dtype=tf.float32)


        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
#        self.state_size = self.units
#        self.output_size = self.units
    
    
    @property
    def state_size(self):
        return [self.units,self.units,1]
    
    @property
    def output_size(self):
        return [self.units * 2, self.units, self.units, 1, 1]  # Must be an integer    
    
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
        po_A_raw = states[0]
        po_B_raw = states[1]
        iteration=states[2]
        perturb_ind=self.pertind


        if self.output_activation is not None:
            po_A = self.output_activation(po_A_raw)
            po_B = self.output_activation(po_B_raw) 

        inputs_0=inputs[:,0:1]# batch_size * input_dim(1)
        inputs_brown=inputs[:,1:]# batch_size * nUnits

        h = backend.dot(inputs_0, self.kernel)
        h_A,h_B=tf.split(h, num_or_size_splits=2, axis=1)
        
        Wr_A, Wr_B, S_A, S_B=tf.split(self.recurrent_kernel,num_or_size_splits=4, axis=1)
        
       
        if self.bias is not None:
            bias_A, bias_B=tf.split(self.bias,num_or_size_splits=2, axis=0)
            h_A += bias_A
            h_B += bias_B

        batch_size = tf.shape(inputs_0)[0]
        
        output_A = h_A + backend.dot(po_A, Wr_A)+backend.dot(po_B, S_B)+tf.random.normal([batch_size,self.units],mean=0,stddev=self.noisesd, seed=self.seed)
        output_B = h_B + backend.dot(po_B, Wr_B)+backend.dot(po_A, S_A)+tf.random.normal([batch_size,self.units],mean=0,stddev=self.noisesd, seed=self.seed)
        
        
        if self.input_activation is not None:
            output_A = self.input_activation(output_A)
            output_B = self.input_activation(output_B)
        
        output_A=po_A_raw+(output_A-po_A_raw)/self.tau
        output_B=po_B_raw+(output_B-po_B_raw)/self.tau
        
        
        iteration=iteration+1
        perturb_ind=tf.cast(perturb_ind,dtype=tf.float32)
        iteration=tf.cast(iteration,dtype=tf.float32)
        boolind=tf.math.equal(iteration,perturb_ind)# perturb_ind is [batch, anynumber]
        
        
        boolind_A=tf.math.reduce_any(tf.math.logical_and(boolind,self.pert_A),axis=1,keepdims=True)
        boolind_B=tf.math.reduce_any(tf.math.logical_and(boolind,self.pert_B),axis=1,keepdims=True)
        
        noise_scale_A=tf.random.normal([batch_size,1],mean=0,stddev=self.pert_noisesd)
        noise_scale_B=tf.random.normal([batch_size,1],mean=0,stddev=self.pert_noisesd)
        if self.sync_noise is True:
            boolind_both=tf.math.logical_or(boolind_A, boolind_B)
            boolind_A=boolind_both
            boolind_B=boolind_both
            noise_scale_B=noise_scale_A
        
        
        noise_add_A=backend.dot(noise_scale_A,self.noise_vec_A)# (batch,units)
        output_A_noise=output_A+noise_add_A
        output_A=tf.where(boolind_A,output_A_noise,output_A) # add noise only if iteration=pertrub_ind

        noise_add_B=backend.dot(noise_scale_B,self.noise_vec_B)# (batch,units)
        output_B_noise=output_B+noise_add_B
        output_B=tf.where(boolind_B,output_B_noise,output_B)


        # add brownian noise
        brown_noise = backend.dot(inputs_brown,self.noise_weights)#(batch,rank),(rank,units)
        output_A += brown_noise # (batch,units)
        output_B += brown_noise # (batch,units)           
        
        
        binary_A=tf.cast(boolind_A, tf.float32)
        binary_B=tf.cast(boolind_B, tf.float32)
        
        if self.output_activation is not None:
            real_output_A = self.output_activation(output_A)
            real_output_B = self.output_activation(output_B) 
        outputs = tf.concat([real_output_A, real_output_B], axis=-1)# Shape: (batch_size, units * 2) 
        return [real_output_A, real_output_B, noise_add_A, noise_add_B, binary_A, binary_B], [output_A,output_B,iteration] # put real output and then states you want to pass to next iteration


    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        po_A_raw = tf.zeros((batch_size, self.units), dtype=dtype)
        po_B_raw = tf.zeros((batch_size, self.units), dtype=dtype)
        iteration = tf.zeros((batch_size, 1), dtype=tf.float32)
        return [po_A_raw, po_B_raw, iteration]




###########################################
# for getting lyapunov exponents

import tensorflow as tf
from tensorflow.keras.layers import AbstractRNNCell
from tensorflow.keras import activations, initializers, regularizers, constraints, backend

class RNNCustom2FixPerturb_noise_prob_brown_lyapunov(AbstractRNNCell):
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
        kernel_trainable=True,  # specify whether to train input kernel or not
        dropout=0.0,
        recurrent_dropout=0.0,
        seed=None,
        tau=10,
        noisesd=0,
        perturb_ind=[0, 0],
        pert_which=[True, True],  # boolean list: if True, perturb RNN A; otherwise, RNN B
        pert_noisesd=0,
        noise_weights=0,  # expected to be a tensor of shape (rank, units)
        **kwargs,
    ):
        if units <= 0:
            raise ValueError(
                "Received an invalid value for argument `units`, "
                f"expected a positive integer, got {units}."
            )
        super().__init__(**kwargs)

        self.units = units
        self.input_activation = activations.get(input_activation)
        self.output_activation = activations.get(output_activation)
        self.use_bias = use_bias
        self.tau = tau
        self.noisesd = noisesd
        self.kernel_trainable = kernel_trainable
        self.seed = seed
        self.pertind = tf.convert_to_tensor(perturb_ind, dtype=tf.int32)
        self.pert_noisesd = pert_noisesd
        self.pert_A = pert_which
        self.pert_B = tf.math.logical_not(pert_which)
        self.noise_weights = tf.cast(noise_weights, dtype=tf.float32)  # should be shape: (rank, units)

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

    @property
    def state_size(self):
        # States: [po_A_raw, po_B_raw, iteration]
        return [self.units, self.units, 1]

    @property
    def output_size(self):
        # We output a list:
        #   outputs_state: concatenated updated state from RNN A and RNN B of shape (batch, 2*units)
        #   jacobian_flat: flattened Jacobian of shape (batch, 4*units^2)
        # Hence, output_size is a list.
        return [self.units * 2, 4 * (self.units ** 2)]

    def build(self, input_shape):
        # Kernel shape is (1, 2*units) because the first component of inputs is a scalar.
        self.kernel = self.add_weight(
            shape=(1, self.units * 2),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=self.kernel_trainable
        )
        # Recurrent kernel is of shape (units, 4*units); later split into four parts.
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units * 2,),
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=False):
        # Unpack current states.
        po_A_raw = states[0]  # shape: (batch, units)
        po_B_raw = states[1]  # shape: (batch, units)
        iteration = states[2]  # shape: (batch, 1)

        # For perturbations.
        perturb_ind = self.pertind

        # Split inputs: first column for primary input; remaining for Brownian noise.
        inputs_0 = inputs[:, 0:1]      # shape: (batch, 1)
        inputs_brown = inputs[:, 1:]   # shape: (batch, remaining)

        # Compute input contribution.
        h = backend.dot(inputs_0, self.kernel)  # shape: (batch, 2*units)
        h_A, h_B = tf.split(h, num_or_size_splits=2, axis=1)  # each is (batch, units)

        # Split recurrent kernel into four parts.
        Wr_A, Wr_B, S_A, S_B = tf.split(self.recurrent_kernel, num_or_size_splits=4, axis=1)

        if self.bias is not None:
            bias_A, bias_B = tf.split(self.bias, num_or_size_splits=2, axis=0)
            h_A += bias_A
            h_B += bias_B

        batch_size = tf.shape(inputs_0)[0]

        # === Compute the Jacobian using GradientTape ===
        # We treat the current combined state as
        # old_state = [po_A_raw, po_B_raw] with shape (batch, 2*units).
        old_state = tf.concat([po_A_raw, po_B_raw], axis=1)
        with tf.GradientTape() as tape:
            tape.watch(old_state)
            # Apply output activation to each state component.
            if self.output_activation is not None:
                po_A = self.output_activation(old_state[:, :self.units])
                po_B = self.output_activation(old_state[:, self.units:])
            else:
                po_A = old_state[:, :self.units]
                po_B = old_state[:, self.units:]
            # Compute the internal update for each part.
            output_A_internal = (h_A +
                                 backend.dot(po_A, Wr_A) +
                                 backend.dot(po_B, S_B) +
                                 tf.random.normal([batch_size, self.units],
                                                   mean=0, stddev=self.noisesd, seed=self.seed))
            output_B_internal = (h_B +
                                 backend.dot(po_B, Wr_B) +
                                 backend.dot(po_A, S_A) +
                                 tf.random.normal([batch_size, self.units],
                                                   mean=0, stddev=self.noisesd, seed=self.seed))
            if self.input_activation is not None:
                output_A_internal = self.input_activation(output_A_internal)
                output_B_internal = self.input_activation(output_B_internal)
            # Euler-style integration to update state.
            new_state_A = old_state[:, :self.units] + (output_A_internal - old_state[:, :self.units]) / self.tau
            new_state_B = old_state[:, self.units:] + (output_B_internal - old_state[:, self.units:]) / self.tau
            new_state = tf.concat([new_state_A, new_state_B], axis=1)  # (batch, 2*units)
        # Calculate the Jacobian matrix: d(new_state)/d(old_state), shape: (batch, 2*units, 2*units).
        jacobian = tape.batch_jacobian(new_state, old_state)
        jacobian_flat = tf.reshape(jacobian, [batch_size, -1])  # shape: (batch, 4*units^2)

        # === End Jacobian computation ===

        # Increase the iteration counter.
        iteration = iteration + 1
        boolind = tf.math.equal(tf.cast(iteration,dtype=tf.int32), perturb_ind)

        boolind_A = tf.math.reduce_any(tf.math.logical_and(boolind, self.pert_A), axis=1, keepdims=True)
        output_A_noise = new_state_A + tf.random.normal([batch_size, self.units],
                                                        mean=0, stddev=self.pert_noisesd, seed=self.seed)
        output_A_final = tf.where(boolind_A, output_A_noise, new_state_A)

        boolind_B = tf.math.reduce_any(tf.math.logical_and(boolind, self.pert_B), axis=1, keepdims=True)
        output_B_noise = new_state_B + tf.random.normal([batch_size, self.units],
                                                        mean=0, stddev=self.pert_noisesd, seed=self.seed)
        output_B_final = tf.where(boolind_B, output_B_noise, new_state_B)

        # Add Brownian noise.
        brown_noise = backend.dot(inputs_brown, self.noise_weights)  # shape: (batch, units)
        output_A_final += brown_noise
        output_B_final += brown_noise

        # Create the outputs as a list:
        # outputs_state: updated state from RNN A and B (shape: (batch, 2*units))
        # jacobian_flat: linearized Jacobian (shape: (batch, 4*units^2))
        #outputs_state = tf.concat([output_A_final, output_B_final], axis=1)
        outputs_state = tf.concat([po_A, po_B], axis=-1)# Shape: (batch_size, units * 2) 
        outputs = [outputs_state, jacobian_flat]

        # Return the outputs and new states.
        return outputs, [output_A_final, output_B_final, iteration]



class RNNCustom2FixPerturb_noise_prob_brown_noise_jacobian(AbstractRNNCell):
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
        kernel_trainable=True,  # specify whether to train input kernel or not
        dropout=0.0,
        recurrent_dropout=0.0,
        seed=None,
        tau=10,
        noisesd=0,
        perturb_ind=[0, 0],
        pert_which=[True, True],  # boolean list: if True, perturb RNN A; otherwise, RNN B
        pert_noisesd=0,
        noise_weights=0,  # expected to be a tensor of shape (rank, units)
        **kwargs,
    ):
        if units <= 0:
            raise ValueError(
                "Received an invalid value for argument `units`, "
                f"expected a positive integer, got {units}."
            )
        super().__init__(**kwargs)

        self.units = units
        self.input_activation = activations.get(input_activation)
        self.output_activation = activations.get(output_activation)
        self.use_bias = use_bias
        self.tau = tau
        self.noisesd = noisesd
        self.kernel_trainable = kernel_trainable
        self.seed = seed
        self.pertind = tf.convert_to_tensor(perturb_ind, dtype=tf.int32)
        self.pert_noisesd = pert_noisesd
        self.pert_A = pert_which
        self.pert_B = tf.math.logical_not(pert_which)
        self.noise_weights = tf.cast(noise_weights, dtype=tf.float32)  # should be shape: (rank, units)
        self.noise_rank=tf.shape(noise_weights)[0]
        
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

    @property
    def state_size(self):
        # States: [po_A_raw, po_B_raw, iteration]
        return [self.units, self.units, 1]

    @property
    def output_size(self):
        # We output a list:
        #   outputs_state: concatenated updated state from RNN A and RNN B of shape (batch, 2*units)
        #   jacobian_flat: flattened Jacobian of shape (batch, 4*units^2)
        # Hence, output_size is a list.
        return [self.units * 2, 4 * (self.units ** 2)]

    def build(self, input_shape):
        # Kernel shape is (1, 2*units) because the first component of inputs is a scalar.
        self.kernel = self.add_weight(
            shape=(1, self.units * 2),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=self.kernel_trainable
        )
        # Recurrent kernel is of shape (units, 4*units); later split into four parts.
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units * 2,),
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=False):
        # Unpack current states.
        po_A_raw0 = states[0]  # shape: (batch, units)
        po_B_raw0 = states[1]  # shape: (batch, units)
        iteration = states[2]  # shape: (batch, 1)

        # For perturbations.
        perturb_ind = self.pertind

        # Split inputs: first column for primary input; remaining for Brownian noise.
        inputs_0 = inputs[:, 0:1]      # shape: (batch, 1)
        inputs_brown = inputs[:, 1:]   # shape: (batch, remaining)

        # Compute input contribution.
        h = backend.dot(inputs_0, self.kernel)  # shape: (batch, 2*units)
        h_A, h_B = tf.split(h, num_or_size_splits=2, axis=1)  # each is (batch, units)

        # Split recurrent kernel into four parts.
        Wr_A, Wr_B, S_A, S_B = tf.split(self.recurrent_kernel, num_or_size_splits=4, axis=1)

        if self.bias is not None:
            bias_A, bias_B = tf.split(self.bias, num_or_size_splits=2, axis=0)
            h_A += bias_A
            h_B += bias_B

        batch_size = tf.shape(inputs_0)[0]

        # === Compute the Jacobian using GradientTape ===
        # We treat the current combined state as
        # old_state = [po_A_raw, po_B_raw] with shape (batch, 2*units).
        # Add Brownian noise.
        brown_noise = backend.dot(inputs_brown, self.noise_weights)  # shape: (batch, units)

        

        with tf.GradientTape() as tape:
            tape.watch(inputs_brown)# (batch, rank=2)
            brown_noise = backend.dot(inputs_brown, self.noise_weights)  # shape: (batch, units)
            po_A_raw = po_A_raw0 + brown_noise
            po_B_raw = po_B_raw0 + brown_noise
            # Apply output activation to each state component.
            if self.output_activation is not None:
                po_A = self.output_activation(po_A_raw)
                po_B = self.output_activation(po_B_raw)
            else:
                po_A = po_A_raw
                po_B = po_B_raw 
            # Compute the internal update for each part.
            output_A_internal = (h_A +
                                 backend.dot(po_A, Wr_A) +
                                 backend.dot(po_B, S_B) +
                                 tf.random.normal([batch_size, self.units],
                                                   mean=0, stddev=self.noisesd, seed=self.seed))
            output_B_internal = (h_B +
                                 backend.dot(po_B, Wr_B) +
                                 backend.dot(po_A, S_A) +
                                 tf.random.normal([batch_size, self.units],
                                                   mean=0, stddev=self.noisesd, seed=self.seed))
            if self.input_activation is not None:
                output_A_internal = self.input_activation(output_A_internal)
                output_B_internal = self.input_activation(output_B_internal)
            # Euler-style integration to update state.
            new_state_A = po_A_raw + (output_A_internal - po_A_raw) / self.tau
            new_state_B = po_B_raw + (output_B_internal - po_B_raw) / self.tau
            new_state = tf.concat([new_state_A, new_state_B], axis=1)  # (batch, 2*units)
        # Calculate the Jacobian matrix: d(new_state)/d(old_state), shape: (batch, 2*units, 2*units).
        jacobian = tape.batch_jacobian(new_state, inputs_brown) # batch, 2*nUnit, 2
        jacobian_flat = tf.reshape(jacobian, [batch_size, -1])  # shape: (batch, 2*units)

        
        brown_noise_sub = tf.stop_gradient(backend.dot(inputs_brown, self.noise_weights))  # shape: (batch, units)
        with tf.GradientTape() as tape:
            tape.watch(inputs_brown)
            brown_noise = backend.dot(inputs_brown, self.noise_weights)  # shape: (batch, units)
            po_A_raw = po_A_raw0 + brown_noise
            po_B_raw = po_B_raw0 + brown_noise
            
            po_A_raw_sub  = po_A_raw0 + brown_noise_sub # brown_noise_sub should act like a constant
            po_B_raw_sub  = po_B_raw0 + brown_noise_sub            
            # Apply output activation to each state component.
            if self.output_activation is not None:
                po_A = self.output_activation(po_A_raw)
                po_B = self.output_activation(po_B_raw)
                po_A_sub  = self.output_activation(po_A_raw_sub)
                po_B_sub  = self.output_activation(po_B_raw_sub)
            else:
                po_A = po_A_raw
                po_B = po_B_raw 
                po_A_sub = po_A_raw_sub
                po_B_sub = po_B_raw_sub 
            # Compute the internal update for each part.
            output_A_internal = (h_A +
                                 backend.dot(po_A, Wr_A) +
                                 backend.dot(po_B_sub, S_B) +
                                 tf.random.normal([batch_size, self.units],
                                                   mean=0, stddev=self.noisesd, seed=self.seed))
            output_B_internal = (h_B +
                                 backend.dot(po_B, Wr_B) +
                                 backend.dot(po_A_sub, S_A) +
                                 tf.random.normal([batch_size, self.units],
                                                   mean=0, stddev=self.noisesd, seed=self.seed))
            if self.input_activation is not None:
                output_A_internal = self.input_activation(output_A_internal)
                output_B_internal = self.input_activation(output_B_internal)
            # Euler-style integration to update state.
            new_state_A_sub = po_A_raw + (output_A_internal - po_A_raw) / self.tau
            new_state_B_sub = po_B_raw + (output_B_internal - po_B_raw) / self.tau
            new_state_sub = tf.concat([new_state_A_sub, new_state_B_sub], axis=1)  # (batch, 2*units)
        # Calculate the Jacobian matrix: d(new_state)/d(old_state), shape: (batch, 2*units, 2*units).
        jacobian_sub = tape.batch_jacobian(new_state_sub, inputs_brown) # batch, 2*nUnit, 2
        jacobian_flat_sub = tf.reshape(jacobian_sub, [batch_size, -1])  # shape: (batch, 2*units*2)



        # === End Jacobian computation ===

        # Increase the iteration counter.
        iteration = iteration + 1
        boolind = tf.math.equal(tf.cast(iteration,dtype=tf.int32), perturb_ind)

        boolind_A = tf.math.reduce_any(tf.math.logical_and(boolind, self.pert_A), axis=1, keepdims=True)
        output_A_noise = new_state_A + tf.random.normal([batch_size, self.units],
                                                        mean=0, stddev=self.pert_noisesd, seed=self.seed)
        output_A_final = tf.where(boolind_A, output_A_noise, new_state_A)

        boolind_B = tf.math.reduce_any(tf.math.logical_and(boolind, self.pert_B), axis=1, keepdims=True)
        output_B_noise = new_state_B + tf.random.normal([batch_size, self.units],
                                                        mean=0, stddev=self.pert_noisesd, seed=self.seed)
        output_B_final = tf.where(boolind_B, output_B_noise, new_state_B)



        # Create the outputs as a list:
        # outputs_state: updated state from RNN A and B (shape: (batch, 2*units))
        # jacobian_flat: linearized Jacobian (shape: (batch, 4*units^2))
        #outputs_state = tf.concat([output_A_final, output_B_final], axis=1)
        outputs_state = tf.concat([po_A, po_B], axis=-1)# Shape: (batch_size, units * 2) 
        outputs = [outputs_state, jacobian_flat, jacobian_flat_sub]

        # Return the outputs and new states.
        return outputs, [output_A_final, output_B_final, iteration]
    
    
class RNNCustom2FixPerturb_noise_prob_brown_noise_jacobian_2(AbstractRNNCell):
    # this takes the derivative of the next state with respect to noise input,
    # but unlike the one above, this noise has specific direction specified by the mat input
    # So this tests whether the system is robust against a noise input along specific direction, and
    # how robust agains simultaneous vs non simultaneous inputs
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
        kernel_trainable=True,  # specify whether to train input kernel or not
        dropout=0.0,
        recurrent_dropout=0.0,
        seed=None,
        tau=10,
        noisesd=0,
        perturb_ind=[0, 0],
        pert_which=[True, True],  # boolean list: if True, perturb RNN A; otherwise, RNN B
        pert_noisesd=0,
        noise_weights=0,  # expected to be a tensor of shape (rank, units)
        noise_mat=None, #(2*512,m) 
        project_mat=None, #(2*512,m)
        **kwargs,
    ):
        if units <= 0:
            raise ValueError(
                "Received an invalid value for argument `units`, "
                f"expected a positive integer, got {units}."
            )
        super().__init__(**kwargs)

        self.units = units
        self.input_activation = activations.get(input_activation)
        self.output_activation = activations.get(output_activation)
        self.use_bias = use_bias
        self.tau = tau
        self.noisesd = noisesd
        self.kernel_trainable = kernel_trainable
        self.seed = seed
        self.pertind = tf.convert_to_tensor(perturb_ind, dtype=tf.int32)
        self.pert_noisesd = pert_noisesd
        self.pert_A = pert_which
        self.pert_B = tf.math.logical_not(pert_which)
        self.noise_weights = tf.cast(noise_weights, dtype=tf.float32)  # should be shape: (rank, units)
        self.noise_rank=tf.shape(noise_weights)[0]
        # Split the stacked noise matrix into two halves:
        noise_splits = tf.split(noise_mat, num_or_size_splits=2, axis=0)   # shapes: (512, m), (512, m)
        # Cast and transpose each to float32:
        self.noise_mat_A = tf.transpose(
            tf.cast(noise_splits[0], dtype=tf.float32)
        )  # shape → (m, 512)
        self.noise_mat_B = tf.transpose(
            tf.cast(noise_splits[1], dtype=tf.float32)
        )  # shape → (m, 512)
        proj_splits = tf.split(project_mat, num_or_size_splits=2, axis=0)  # shapes: (512, m), (512, m)
        self.proj_mat_A = tf.transpose(
            tf.cast(proj_splits[0], dtype=tf.float32)
        )  # shape → (m, 512)
        self.proj_mat_B = tf.transpose(
            tf.cast(proj_splits[1], dtype=tf.float32)
        )  # shape → (m, 512)
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

    @property
    def state_size(self):
        # States: [po_A_raw, po_B_raw, iteration]
        return [self.units, self.units, 1]

    @property
    def output_size(self):
        # We output a list:
        #   outputs_state: concatenated updated state from RNN A and RNN B of shape (batch, 2*units)
        #   jacobian_flat: flattened Jacobian of shape (batch, 4*units^2)
        # Hence, output_size is a list.
        return [self.units * 2, 4 * (self.units ** 2)]

    def build(self, input_shape):
        # Kernel shape is (1, 2*units) because the first component of inputs is a scalar.
        self.kernel = self.add_weight(
            shape=(1, self.units * 2),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=self.kernel_trainable
        )
        # Recurrent kernel is of shape (units, 4*units); later split into four parts.
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units * 2,),
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=False):
        # Unpack current states.
        po_A_raw0 = states[0]  # shape: (batch, units)
        po_B_raw0 = states[1]  # shape: (batch, units)
        iteration = states[2]  # shape: (batch, 1)

        # For perturbations.
        perturb_ind = self.pertind

        # Split inputs: first column for primary input; remaining for Brownian noise.
        inputs_0 = inputs[:, 0:1]      # shape: (batch, 1)
        inputs_brown = inputs[:, 1:]   # shape: (batch, remaining)

        # Compute input contribution.
        h = backend.dot(inputs_0, self.kernel)  # shape: (batch, 2*units)
        h_A, h_B = tf.split(h, num_or_size_splits=2, axis=1)  # each is (batch, units)

        # Split recurrent kernel into four parts.
        Wr_A, Wr_B, S_A, S_B = tf.split(self.recurrent_kernel, num_or_size_splits=4, axis=1)

        if self.bias is not None:
            bias_A, bias_B = tf.split(self.bias, num_or_size_splits=2, axis=0)
            h_A += bias_A
            h_B += bias_B

        batch_size = tf.shape(inputs_0)[0]

        # === Compute the Jacobian using GradientTape ===
        # We treat the current combined state as
        # old_state = [po_A_raw, po_B_raw] with shape (batch, 2*units).
        # Add Brownian noise.
        brown_noise = backend.dot(inputs_brown, self.noise_weights)  # shape: (batch, units)

        
        noise_dim = self.noise_mat_A.shape[0]            # int
        noise_scale=tf.zeros([batch_size,noise_dim],dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(noise_scale)#
            noise_add_A=backend.dot(noise_scale,self.noise_mat_A) 
            noise_add_B=backend.dot(noise_scale,self.noise_mat_B)
            po_A_raw = po_A_raw0 + brown_noise + noise_add_A # batch, nUnit
            po_B_raw = po_B_raw0 + brown_noise + noise_add_B
            # Apply output activation to each state component.
            if self.output_activation is not None:
                po_A = self.output_activation(po_A_raw)
                po_B = self.output_activation(po_B_raw)
            else:
                po_A = po_A_raw
                po_B = po_B_raw 
            # Compute the internal update for each part.
            output_A_internal = (h_A +
                                 backend.dot(po_A, Wr_A) +
                                 backend.dot(po_B, S_B) +
                                 tf.random.normal([batch_size, self.units],
                                                   mean=0, stddev=self.noisesd, seed=self.seed))
            output_B_internal = (h_B +
                                 backend.dot(po_B, Wr_B) +
                                 backend.dot(po_A, S_A) +
                                 tf.random.normal([batch_size, self.units],
                                                   mean=0, stddev=self.noisesd, seed=self.seed))
            if self.input_activation is not None:
                output_A_internal = self.input_activation(output_A_internal)
                output_B_internal = self.input_activation(output_B_internal)
            # Euler-style integration to update state.
            new_state_A = po_A_raw + (output_A_internal - po_A_raw) / self.tau # (batch, units)
            new_state_B = po_B_raw + (output_B_internal - po_B_raw) / self.tau
            new_state = tf.concat([new_state_A, new_state_B], axis=1)  # (batch, 2*units)
            
            # project the result to the cca axis
            proj_A = tf.linalg.matmul(new_state_A, tf.transpose(self.proj_mat_A))# (batch,nUnit)*(nUnit,m)->(batch,m)
            proj_B = tf.linalg.matmul(new_state_B, tf.transpose(self.proj_mat_B))# (batch,nUnit)*(nUnit,m)->(batch,m)
        
        # Calculate the Jacobian matrix: d(new_state)/d(old_state), shape: (batch, 2*units, 2*units).
        jacobian_A = tape.batch_jacobian(proj_A, noise_scale) # batch, m, m
        jacobian_B = tape.batch_jacobian(proj_B, noise_scale) # batch, m, m
        del tape
        
        jacobian_A_flat = tf.reshape(jacobian_A, [batch_size, -1])  # shape: (batch, m*m)
        jacobian_B_flat = tf.reshape(jacobian_B, [batch_size, -1])  # shape: (batch, m*m)

        
        brown_noise_sub =backend.dot(inputs_brown, self.noise_weights)  # shape: (batch, units)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(noise_scale)#
            noise_add_A=backend.dot(noise_scale,self.noise_mat_A) 
            noise_add_B=backend.dot(noise_scale,self.noise_mat_B)
            po_A_raw = po_A_raw0 + brown_noise + noise_add_A # batch, nUnit
            po_B_raw = po_B_raw0 + brown_noise + noise_add_B
            
            po_A_raw_sub  = po_A_raw0 + brown_noise # brown_noise_sub should act like a constant
            po_B_raw_sub  = po_B_raw0 + brown_noise            
            # Apply output activation to each state component.
            if self.output_activation is not None:
                po_A = self.output_activation(po_A_raw)
                po_B = self.output_activation(po_B_raw)
                po_A_sub  = self.output_activation(po_A_raw_sub)
                po_B_sub  = self.output_activation(po_B_raw_sub)
            else:
                po_A = po_A_raw
                po_B = po_B_raw 
                po_A_sub = po_A_raw_sub
                po_B_sub = po_B_raw_sub 
            # Compute the internal update for each part.
            output_A_internal = (h_A +
                                 backend.dot(po_A, Wr_A) +
                                 backend.dot(po_B_sub, S_B) +
                                 tf.random.normal([batch_size, self.units],
                                                   mean=0, stddev=self.noisesd, seed=self.seed))
            output_B_internal = (h_B +
                                 backend.dot(po_B, Wr_B) +
                                 backend.dot(po_A_sub, S_A) +
                                 tf.random.normal([batch_size, self.units],
                                                   mean=0, stddev=self.noisesd, seed=self.seed))
            if self.input_activation is not None:
                output_A_internal = self.input_activation(output_A_internal)
                output_B_internal = self.input_activation(output_B_internal)
            # Euler-style integration to update state.
            new_state_A_sub = po_A_raw + (output_A_internal - po_A_raw) / self.tau
            new_state_B_sub = po_B_raw + (output_B_internal - po_B_raw) / self.tau
            new_state_sub = tf.concat([new_state_A_sub, new_state_B_sub], axis=1)  # (batch, 2*units)

            # project the result to the cca axis
            proj_A_sub = tf.linalg.matmul(new_state_A_sub, tf.transpose(self.proj_mat_A))# (batch,nUnit)*(nUnit,m)->(batch,m)
            proj_B_sub = tf.linalg.matmul(new_state_B_sub, tf.transpose(self.proj_mat_B))# (batch,nUnit)*(nUnit,m)->(batch,m)
        
        # Calculate the Jacobian matrix: d(new_state)/d(old_state), shape: (batch, 2*units, 2*units).
        jacobian_A_sub = tape.batch_jacobian(proj_A_sub, noise_scale) # batch, m, m
        jacobian_B_sub = tape.batch_jacobian(proj_B_sub, noise_scale) # batch, m, m
        del tape
        jacobian_A_flat_sub = tf.reshape(jacobian_A_sub, [batch_size, -1])  # shape: (batch, m*m)
        jacobian_B_flat_sub = tf.reshape(jacobian_B_sub, [batch_size, -1])  # shape: (batch, m*m)


        # === End Jacobian computation ===

        # Increase the iteration counter.
        iteration = iteration + 1
        boolind = tf.math.equal(tf.cast(iteration,dtype=tf.int32), perturb_ind)

        boolind_A = tf.math.reduce_any(tf.math.logical_and(boolind, self.pert_A), axis=1, keepdims=True)
        output_A_noise = new_state_A + tf.random.normal([batch_size, self.units],
                                                        mean=0, stddev=self.pert_noisesd, seed=self.seed)
        output_A_final = tf.where(boolind_A, output_A_noise, new_state_A)

        boolind_B = tf.math.reduce_any(tf.math.logical_and(boolind, self.pert_B), axis=1, keepdims=True)
        output_B_noise = new_state_B + tf.random.normal([batch_size, self.units],
                                                        mean=0, stddev=self.pert_noisesd, seed=self.seed)
        output_B_final = tf.where(boolind_B, output_B_noise, new_state_B)



        # Create the outputs as a list:
        # outputs_state: updated state from RNN A and B (shape: (batch, 2*units))
        # jacobian_flat: linearized Jacobian (shape: (batch, 4*units^2))
        #outputs_state = tf.concat([output_A_final, output_B_final], axis=1)
        outputs_state = tf.concat([po_A, po_B], axis=-1)# Shape: (batch_size, units * 2) 
        outputs = [outputs_state, 
                   jacobian_A_flat, #(batch, m*m)
                   jacobian_B_flat, #(batch, m*m)
                   jacobian_A_flat_sub,  #(batch, m*m)
                   jacobian_B_flat_sub] #(batch, m*m)

        # Return the outputs and new states.
        return outputs, [output_A_final, output_B_final, iteration]    
    
    
    
class RNNCustom2FixPerturb_noise_prob_synch_jacobian(AbstractRNNCell):
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
        kernel_trainable=True,  # specify whether to train input kernel or not
        dropout=0.0,
        recurrent_dropout=0.0,
        seed=None,
        tau=10,
        noisesd=0,
        perturb_ind=[0, 0],
        pert_which=[True, True],  # boolean list: if True, perturb RNN A; otherwise, RNN B
        pert_noisesd=0,
        noise_weights=0,  # expected to be a tensor of shape (rank, units)
        lr_AB=[None,None],
        lr_BA=[None,None],
        jacob_at_zero=False,
        **kwargs,
    ):
        if units <= 0:
            raise ValueError(
                "Received an invalid value for argument `units`, "
                f"expected a positive integer, got {units}."
            )
        super().__init__(**kwargs)

        self.units = units
        self.input_activation = activations.get(input_activation)
        self.output_activation = activations.get(output_activation)
        self.use_bias = use_bias
        self.tau = tau
        self.noisesd = noisesd
        self.kernel_trainable = kernel_trainable
        self.seed = seed
        self.pertind = tf.convert_to_tensor(perturb_ind, dtype=tf.int32)
        self.pert_noisesd = pert_noisesd
        self.pert_A = pert_which
        self.pert_B = tf.math.logical_not(pert_which)
        self.noise_weights = tf.cast(noise_weights, dtype=tf.float32)  # should be shape: (rank, units)

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
        
        # load linear regression weihgts
        self.lr_AB_weight, self.lr_AB_bias=lr_AB# (nUnits, nUnits), (nUnits,)
        self.lr_BA_weight, self.lr_BA_bias=lr_BA
        
        self.lr_AB_weight=tf.constant(self.lr_AB_weight, dtype=tf.float32)
        self.lr_BA_weight=tf.constant(self.lr_BA_weight, dtype=tf.float32)
        self.lr_AB_bias=tf.constant(self.lr_AB_bias, dtype=tf.float32)
        self.lr_BA_bias=tf.constant(self.lr_BA_bias, dtype=tf.float32)
        
        self.lr_AB_bias=tf.reshape(self.lr_AB_bias,[1,-1]) #(1, nUnits)
        self.lr_BA_bias=tf.reshape(self.lr_BA_bias,[1,-1]) #(1, nUnits)
        
        
        self.jacob_at_zero=jacob_at_zero
        
    @property
    def state_size(self):
        # States: [po_A_raw, po_B_raw, iteration]
        return [self.units, self.units, 1]

    @property
    def output_size(self):
        # We output a list:
        #   outputs_state: concatenated updated state from RNN A and RNN B of shape (batch, 2*units)
        #   jacobian_flat: flattened Jacobian of shape (batch, 4*units^2)
        # Hence, output_size is a list.
        return [self.units * 2, 4 * (self.units ** 2)]

    def build(self, input_shape):
        # Kernel shape is (1, 2*units) because the first component of inputs is a scalar.
        self.kernel = self.add_weight(
            shape=(1, self.units * 2),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=self.kernel_trainable
        )
        # Recurrent kernel is of shape (units, 4*units); later split into four parts.
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units * 2,),
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=False):
        # Unpack current states.
        po_A_raw = states[0]  # shape: (batch, units)
        po_B_raw = states[1]  # shape: (batch, units)
        iteration = states[2]  # shape: (batch, 1)

        # For perturbations.
        perturb_ind = self.pertind

        # Split inputs: first column for primary input; remaining for Brownian noise.
        inputs_0 = inputs[:, 0:1]      # shape: (batch, 1)
        inputs_brown = inputs[:, 1:]   # shape: (batch, remaining)

        # Compute input contribution.
        h = backend.dot(inputs_0, self.kernel)  # shape: (batch, 2*units)
        h_A, h_B = tf.split(h, num_or_size_splits=2, axis=1)  # each is (batch, units)

        # Split recurrent kernel into four parts.
        Wr_A, Wr_B, S_A, S_B = tf.split(self.recurrent_kernel, num_or_size_splits=4, axis=1)

        if self.bias is not None:
            bias_A, bias_B = tf.split(self.bias, num_or_size_splits=2, axis=0)
            h_A += bias_A
            h_B += bias_B

        batch_size = tf.shape(inputs_0)[0]

        # === Compute the Jacobian using GradientTape ===
        # We treat the current combined state as
        # old_state = [po_A_raw, po_B_raw] with shape (batch, 2*units).

        # compute random noise
        noise_A=tf.random.normal([batch_size, self.units],
                                 mean=0, 
                                 stddev=self.noisesd, 
                                 seed=self.seed)
        noise_B=tf.random.normal([batch_size, self.units],
                                 mean=0, 
                                 stddev=self.noisesd, 
                                 seed=self.seed)

        
        # compute offset
        B_pred=tf.matmul(po_A_raw,self.lr_AB_weight) + self.lr_AB_bias # shape: (batch, units)
        A_pred=tf.matmul(po_B_raw,self.lr_BA_weight) + self.lr_BA_bias # shape: (batch, units)
        
        delta_B=tf.stop_gradient(po_B_raw-B_pred)
        delta_A=tf.stop_gradient(po_A_raw-A_pred)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch((delta_A, delta_B))
            
            # Apply output activation to each state component.
            if self.output_activation is not None:
                po_A = self.output_activation(po_A_raw)
                po_B = self.output_activation(po_B_raw)       
                po_A_pred = self.output_activation(A_pred+delta_A)
                po_B_pred = self.output_activation(B_pred+delta_B)
            else:
                po_A = po_A_raw
                po_B = po_B_raw
                po_A_pred=A_pred+delta_A
                po_B_pred=B_pred+delta_B

            # Compute the internal update for each part.
            output_A_internal_all_A = (h_A +
                                 backend.dot(po_A, Wr_A) +
                                 backend.dot(po_B_pred, S_B) +
                                 noise_A)
            output_B_internal_all_A = (h_B +
                                 backend.dot(po_B_pred, Wr_B) +
                                 backend.dot(po_A, S_A) +
                                 noise_B)
            output_A_internal_all_B = (h_A +
                                 backend.dot(po_A_pred, Wr_A) +
                                 backend.dot(po_B, S_B) +
                                 noise_A)
            output_B_internal_all_B = (h_B +
                                 backend.dot(po_B, Wr_B) +
                                 backend.dot(po_A_pred, S_A) +
                                 noise_B)
            if self.input_activation is not None:
                output_A_internal_all_A = self.input_activation(output_A_internal_all_A)
                output_B_internal_all_A = self.input_activation(output_B_internal_all_A)
                output_A_internal_all_B = self.input_activation(output_A_internal_all_B)
                output_B_internal_all_B = self.input_activation(output_B_internal_all_B)
                
                
            # Euler-style integration to update state.
            new_state_A = po_A_raw + (output_A_internal_all_A - po_A_raw) / self.tau
            new_state_B = po_B_raw + (output_B_internal_all_A - po_B_raw) / self.tau
            new_state = tf.concat([new_state_A, new_state_B], axis=1)  # (batch, 2*units)
        
            # calculate the update for the offset
            delta_A_next=((1-1/self.tau)*delta_A 
                          - (1/self.tau)*self.lr_BA_bias
                          +(1/self.tau)*output_A_internal_all_B
                          -(1/self.tau)*tf.matmul(output_B_internal_all_B ,self.lr_BA_weight))
            
            delta_B_next=((1-1/self.tau)*delta_B 
                          - (1/self.tau)*self.lr_AB_bias
                          +(1/self.tau)*output_B_internal_all_A
                          -(1/self.tau)*tf.matmul(output_A_internal_all_A ,self.lr_AB_weight))
        
        # Calculate the Jacobian matrix: d(new_state)/d(old_state), shape: (batch, 2*units, 2*units).
        jacobian_delta_A = tape.batch_jacobian(delta_A_next, delta_A)
        jacobian_delta_B = tape.batch_jacobian(delta_B_next, delta_B)
        del tape                 # free graph ASAP
        jacobian_delta_A_flat = tf.reshape(jacobian_delta_A, [batch_size, -1])  # shape: (batch, 4*units^2)
        jacobian_delta_B_flat = tf.reshape(jacobian_delta_B, [batch_size, -1])  # shape: (batch, 4*units^2)

        # === End Jacobian computation ===


        #===========
        # adding random noise perturbation (we usually do not add the perturbation)
        # Increase the iteration counter.
        iteration = iteration + 1
        boolind = tf.math.equal(tf.cast(iteration,dtype=tf.int32), perturb_ind)

        boolind_A = tf.math.reduce_any(tf.math.logical_and(boolind, self.pert_A), axis=1, keepdims=True)
        output_A_noise = new_state_A + tf.random.normal([batch_size, self.units],
                                                        mean=0, stddev=self.pert_noisesd, seed=self.seed)
        output_A_final = tf.where(boolind_A, output_A_noise, new_state_A)

        boolind_B = tf.math.reduce_any(tf.math.logical_and(boolind, self.pert_B), axis=1, keepdims=True)
        output_B_noise = new_state_B + tf.random.normal([batch_size, self.units],
                                                        mean=0, stddev=self.pert_noisesd, seed=self.seed)
        output_B_final = tf.where(boolind_B, output_B_noise, new_state_B)


        #========
        # Add Brownian noise. (Some models are tained under shared noise inputs)
        brown_noise = backend.dot(inputs_brown, self.noise_weights)  # shape: (batch, units)
        output_A_final += brown_noise
        output_B_final += brown_noise

        # Create the outputs as a list:
        # outputs_state: updated state from RNN A and B (shape: (batch, 2*units))
        # jacobian_flat: linearized Jacobian (shape: (batch, 4*units^2))
        #outputs_state = tf.concat([output_A_final, output_B_final], axis=1)
        outputs_state = tf.concat([po_A, po_B], axis=-1)# Shape: (batch_size, units * 2) 
        outputs = [outputs_state, jacobian_delta_A_flat, jacobian_delta_B_flat]

        # Return the outputs and new states.
        return outputs, [output_A_final, output_B_final, iteration]


from tensorflow.keras.layers import Layer, Dense
class RNNCustom2FixPerturb_noise_prob_brown_noise_jacobian_output(AbstractRNNCell):
    # calculate until the output in the rnn and take the derivative with respect to the noise
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
        kernel_trainable=True,  # specify whether to train input kernel or not
        dropout=0.0,
        recurrent_dropout=0.0,
        seed=None,
        tau=10,
        noisesd=0,
        perturb_ind=[0, 0],
        pert_which=[True, True],  # boolean list: if True, perturb RNN A; otherwise, RNN B
        pert_noisesd=0,
        noise_weights=0,  # expected to be a tensor of shape (rank, units)
        final_output_activation="tanh",
        **kwargs,
    ):
        if units <= 0:
            raise ValueError(
                "Received an invalid value for argument `units`, "
                f"expected a positive integer, got {units}."
            )
        super().__init__(**kwargs)

        self.units = units
        self.input_activation = activations.get(input_activation)
        self.output_activation = activations.get(output_activation)
        self.use_bias = use_bias
        self.tau = tau
        self.noisesd = noisesd
        self.kernel_trainable = kernel_trainable
        self.seed = seed
        self.pertind = tf.convert_to_tensor(perturb_ind, dtype=tf.int32)
        self.pert_noisesd = pert_noisesd
        self.pert_A = pert_which
        self.pert_B = tf.math.logical_not(pert_which)
        self.noise_weights = tf.cast(noise_weights, dtype=tf.float32)  # should be shape: (rank, units)
        self.noise_rank=tf.shape(noise_weights)[0]
        
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

        self._output_dense1 = Dense(
            2,                            # ← output size = 2
            activation=final_output_activation,
            name="output_transform_dense1"
        )
        self._output_dense2 = Dense(
            2,                            # ← output size = 2
            activation=final_output_activation,
            name="output_transform_dense2"
        )
    @property
    def state_size(self):
        # States: [po_A_raw, po_B_raw, iteration]
        return [self.units, self.units, 1]

    @property
    def output_size(self):
        # We output a list:
        #   outputs_state: concatenated updated state from RNN A and RNN B of shape (batch, 2*units)
        #   jacobian_flat: flattened Jacobian of shape (batch, 4*units^2)
        # Hence, output_size is a list.
        return [4, 8]

    def build(self, input_shape):
        # Kernel shape is (1, 2*units) because the first component of inputs is a scalar.
        self.kernel = self.add_weight(
            shape=(1, self.units * 2),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=self.kernel_trainable
        )
        # Recurrent kernel is of shape (units, 4*units); later split into four parts.
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units * 2,),
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
            
        
        self.built = True

    def call(self, inputs, states, training=False):
        # Unpack current states.
        po_A_raw0 = states[0]  # shape: (batch, units)
        po_B_raw0 = states[1]  # shape: (batch, units)
        iteration = states[2]  # shape: (batch, 1)

        # For perturbations.
        perturb_ind = self.pertind

        # Split inputs: first column for primary input; remaining for Brownian noise.
        inputs_0 = inputs[:, 0:1]      # shape: (batch, 1)
        inputs_brown = inputs[:, 1:]   # shape: (batch, remaining)

        # Compute input contribution.
        h = backend.dot(inputs_0, self.kernel)  # shape: (batch, 2*units)
        h_A, h_B = tf.split(h, num_or_size_splits=2, axis=1)  # each is (batch, units)

        # Split recurrent kernel into four parts.
        Wr_A, Wr_B, S_A, S_B = tf.split(self.recurrent_kernel, num_or_size_splits=4, axis=1)

        if self.bias is not None:
            bias_A, bias_B = tf.split(self.bias, num_or_size_splits=2, axis=0)
            h_A += bias_A
            h_B += bias_B

        batch_size = tf.shape(inputs_0)[0]

        # === Compute the Jacobian using GradientTape ===
        # We treat the current combined state as
        # old_state = [po_A_raw, po_B_raw] with shape (batch, 2*units).
        # Add Brownian noise.
        brown_noise = backend.dot(inputs_brown, self.noise_weights)  # shape: (batch, units)

        

        with tf.GradientTape() as tape:
            tape.watch(inputs_brown)# (batch, rank=2)
            brown_noise = backend.dot(inputs_brown, self.noise_weights)  # shape: (batch, units)
            po_A_raw = po_A_raw0 + brown_noise
            po_B_raw = po_B_raw0 + brown_noise
            # Apply output activation to each state component.
            if self.output_activation is not None:
                po_A = self.output_activation(po_A_raw)
                po_B = self.output_activation(po_B_raw)
            else:
                po_A = po_A_raw
                po_B = po_B_raw 
            # apply dense layer
            dense_out1=self._output_dense1(po_A) 
            dense_out2=self._output_dense2(po_B) 
            
            dense_out= tf.concat([dense_out1,dense_out2], axis=-1)# shape: (batch, 4)
        
        
        # Calculate the Jacobian matrix: d(dense_out(batch,4))/d(inputs_brown(batch,2)), shape: (batch, 4,2).
        jacobian = tape.batch_jacobian(dense_out, inputs_brown) # batch,4, 2
        jacobian_flat = tf.reshape(jacobian, [batch_size, -1])  # shape: (batch, 8)
                    
        
        # Compute the internal update for each part.
        output_A_internal = (h_A +
                             backend.dot(po_A, Wr_A) +
                             backend.dot(po_B, S_B) +
                             tf.random.normal([batch_size, self.units],
                                               mean=0, stddev=self.noisesd, seed=self.seed))
        output_B_internal = (h_B +
                             backend.dot(po_B, Wr_B) +
                             backend.dot(po_A, S_A) +
                             tf.random.normal([batch_size, self.units],
                                               mean=0, stddev=self.noisesd, seed=self.seed))
        if self.input_activation is not None:
            output_A_internal = self.input_activation(output_A_internal)
            output_B_internal = self.input_activation(output_B_internal)
        # Euler-style integration to update state.
        new_state_A = po_A_raw + (output_A_internal - po_A_raw) / self.tau
        new_state_B = po_B_raw + (output_B_internal - po_B_raw) / self.tau
        
        # Increase the iteration counter.
        iteration = iteration + 1
        boolind = tf.math.equal(tf.cast(iteration,dtype=tf.int32), perturb_ind)

        boolind_A = tf.math.reduce_any(tf.math.logical_and(boolind, self.pert_A), axis=1, keepdims=True)
        output_A_noise = new_state_A + tf.random.normal([batch_size, self.units],
                                                        mean=0, stddev=self.pert_noisesd, seed=self.seed)
        output_A_final = tf.where(boolind_A, output_A_noise, new_state_A)

        boolind_B = tf.math.reduce_any(tf.math.logical_and(boolind, self.pert_B), axis=1, keepdims=True)
        output_B_noise = new_state_B + tf.random.normal([batch_size, self.units],
                                                        mean=0, stddev=self.pert_noisesd, seed=self.seed)
        output_B_final = tf.where(boolind_B, output_B_noise, new_state_B)




        # Create the outputs as a list:
        # outputs_state: updated state from RNN A and B (shape: (batch, 2*units))
        # jacobian_flat: linearized Jacobian (shape: (batch, 4*units^2))
        #outputs_state = tf.concat([output_A_final, output_B_final], axis=1)
        outputs = [dense_out, jacobian_flat]

        # Return the outputs and new states.
        return outputs, [output_A_final, output_B_final, iteration]
    
    