import tensorflow as tf
import numpy as np
from keras import layers
from modules.DotProductAttention import DotProductAttention

class MultiHeadAttention(layers.Layer):
    """
    Multi-Head Attention layers

    Parameters
    -
    heads : int
        Number of heads. `d_k` must be divisible by `heads`
    d_k : int
    d_v : int
    d_model : int
        Final layer unit value
    
    """
    def __init__(self, heads, d_k, d_v, d_model, **kwargs):
        super().__init__(**kwargs)
        self.heads = heads
        self.d_k = d_k
        self.attention = DotProductAttention()
        self.W_q = layers.Dense(d_k)
        self.W_k = layers.Dense(d_k)
        self.W_v = layers.Dense(d_v)
        self.W_o = layers.Dense(d_model)

    def reshape_tensor(self, x, heads, flag):
        """
        Convert tensor shape into multi-head array and reverse
        """
        if flag:
            x = tf.reshape(x, shape=(x.shape[0], x.shape[1], heads, -1))
            x = tf.transpose(x, perm=(0, 2, 1, 3))
        else:
            x = tf.transpose(x, perm=(0, 2, 1, 3))
            x = tf.reshape(x, shape=(x.shape[0], x.shape[1], -1))

        return x
        
    
    def call(self, queries, keys, values, mask=None):
        # Split queries, keys, values into multihead
        queries = self.reshape_tensor(self.W_q(queries), self.heads, True)
        # Output shape: (batch_size, heads, seq_len, -1)

        keys = self.reshape_tensor(self.W_k(keys), self.heads, True)
        # Output shape: (batch_size, heads, seq_len, -1)

        values = self.reshape_tensor(self.W_v(values), self.heads, True)
        # Output shape: (batch_size, heads, seq_len, -1)

        output = self.attention(queries, keys, values, d_k=self.d_k, mask=mask)
        # Output shape: (batch_size, heads, seq_len, -1)

        # Concat output heads
        output = self.reshape_tensor(output, self.heads, False)
        
        return self.W_o(output)
