import tensorflow as tf
import numpy as np
from keras import layers

class DotProductAttention(layers.Layer):
    """
    Dot-product Attention layer base on `Attention is all you need`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, queries, keys, values, d_k : int, mask=None):
        """
        Calculate Dot-product Attention
        
        Parameters
        -
        queries : array_like
        keys : array_like
        values : array_like
        d_k: int
        mask: array_like | None
        """
        
        scores = tf.matmul(queries, keys, transpose_b=True) / np.sqrt(d_k)

        if mask is not None:
            scores += -1e9 * mask

        weights = tf.math.softmax(scores)

        return tf.matmul(weights, values)