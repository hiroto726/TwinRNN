# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 23:23:10 2024

@author: Hiroto Imamura
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer

class JacobianLayer(tf.keras.layers.Layer):
    def __init__(self, model, **kwargs):
        super(JacobianLayer, self).__init__(**kwargs)
        self.model = model

    @tf.function
    def call(self, inputs):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            outputs = self.model(inputs)
        jacobian_matrix = tape.batch_jacobian(outputs, inputs)
        return jacobian_matrix
    

class QRDecomposition(tf.keras.layers.Layer):
    def __init__(self,**kwargs)

