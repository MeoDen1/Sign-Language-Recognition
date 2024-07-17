import tensorflow as tf
import numpy as np
from keras import layers

class DotProductAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, queries, keys, values, d_k, mask=None):
        scores = tf.matmul(queries, keys, transpose_b=True) / tf.sqrt(d_k)

        if mask is not None:
            scores += -1e9 * mask

        weights = tf.math.softmax(scores)

        return tf.matmul(weights, values)
    

class MultiHeadAttention():
    def __init__(self, heads, d_k, d_v, d_model, **kwargs):
        super().__init__(**kwargs)
        self.heads = heads
        self.attention = DotProductAttention()
        self.d_k = d_k
        self.W_q = layers.Dense(d_k)
        self.W_k = layers.Dense(d_k)
        self.W_v = layers.Dense(d_v)
        self.W_o = layers.Dense(d_model)

    def reshape_tensor(self, x, heads, flag):
        if flag:
            x = tf.reshape(x, shape=(x.shape[0], x.shape[1], heads, -1))
            x = tf.transpose(x, perm=(0, 2, 1, 3))
        else:
            x = tf.transpose(x, perm=(0, 2, 1, 3))
            x = tf.reshape(x, shape=(x.shape[0], x.shape[1], -1))

        return x
    
    def call(self, queries, keys, values, mask=None):
        # Split queries, keys, values into multihead
        queries = self.reshape_tensor(queries, self.heads, True)
        # Output shape: (batch_size, heads, seq_len, -1)

        keys = self.reshape_tensor(keys, self.heads, True)
        # Output shape: (batch_size, heads, seq_len, -1)

        values = self.reshape_tensor(values, self.heads, True)
        # Output shape: (batch_size, heads, seq_len, -1)

        output = self.attention(queries, keys, values, self.d_k, mask)
        # Output shape: (batch_size, heads, seq_len, -1)

        # Concat output heads
        output = self.reshape_tensor(output, self.heads, False)
        
        return self.W_o(output)
