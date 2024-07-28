import tensorflow as tf
import numpy as np
from keras import layers, models

class AddNormalization(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = layers.LayerNormalization()

    def call(self, x_1, x_2):
        return self.layer_norm(x_1 + x_2)
    

class FeedForward(layers.Layer):
    """
    Feed forward layer with 2 connected NN layers
    """
    def __init__(self, d_ff, d_model, **kwargs):
        super().__init__(**kwargs)
        self.W_1 = layers.Dense(d_ff, activation='relu')
        self.W_2 = layers.Dense(d_model)

    def call(self, x):
        x = self.W_1(x)

        return self.W_2(x)