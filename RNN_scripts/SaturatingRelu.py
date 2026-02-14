# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 18:55:03 2025

@author: Hiroto
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
def custom_saturating_relu_fn(x, alpha=0.0, max_value=200.0):
    """Piecewise saturating ReLU with a small slope beyond max_value."""
    # Region 1: x <= 0
    negative_part = alpha * x
    
    # Region 2: 0 < x <= max_value
    middle_part = x
    
    # Region 3: x > max_value
    beyond_part = max_value + max_value*tf.math.tanh((x - max_value)/max_value)
    
    # Combine with tf.where
    return tf.where(
        x <= 0.0,
        negative_part,
        tf.where(x <= max_value, middle_part, beyond_part)
    )


def custom_log_saturate_relu_fn(x, alpha=0.0, max_value=200.0):
    """Piecewise saturating ReLU with a small slope beyond max_value."""
    # Region 1: x <= 0
    negative_part = alpha * x
    
    # Region 2: 0 < x <= max_value
    middle_part = x
    
    # Region 3: x > max_value
    safe_x = tf.clip_by_value(x, max_value, float('inf'))  # Avoids log(0)
    beyond_part = max_value + max_value*tf.math.log(safe_x/max_value)
    
    # Combine with tf.where
    return tf.where(
        x <= 0.0,
        negative_part,
        tf.where(x <= max_value, middle_part, beyond_part)
    )

@tf.function
class CustomSaturatingReLU(Layer):
    """Keras layer wrapper around the custom saturating ReLU function."""
    def __init__(self, alpha=0.0, max_value=200.0,  **kwargs):
        super().__init__(**kwargs)
        self.max_value = max_value
        self.alpha=alpha

    def call(self, inputs):
        return custom_saturating_relu_fn(inputs, self.alpha, self.max_value)
    
    
    
    
#%% visualize activations
"""
import numpy as np
import matplotlib.pyplot as plt
max_value=200
x_values = np.linspace(-max_value,max_value*10,100)   
alpha=0.5     
fig, ax = plt.subplots()        
y_values = custom_saturating_relu_fn(tf.constant(x_values, dtype=tf.float32), alpha=alpha, max_value=max_value).numpy()
ax.axvline(max_value, color='k', linestyle='--', linewidth=1)
ax.plot(x_values, y_values, label=f"α={alpha}")
ax.axvline(0, color='k', linestyle='--', linewidth=1)
ax.legend()
ax.set_title(f"α={alpha}, slope={slope}")
ax.set_xlabel("Input x")
ax.set_ylabel("Output f(x)")
"""