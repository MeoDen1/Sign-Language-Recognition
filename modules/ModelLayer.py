import tensorflow as tf
import numpy as np
from keras import layers, models

class AddNormalization(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = layers.LayerNormalization()

    def call(self, input_1, input_2):
        return self.layer_norm(input_1 +input_2)
    

class FeedForward(layers.Layer):
    def __init__(self, d_ff, d_model, **kwargs):
        self.W_1 = layers.Dense(d_ff, activation='relu')
        self.W_2 = layers.Dense(d_model)

    def call(self, input):
        x = self.W_1(input)

        return self.W_2(x)