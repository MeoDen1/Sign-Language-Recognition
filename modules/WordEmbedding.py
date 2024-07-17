import tensorflow as tf
import numpy as np
from keras import layers

class WordEmbedding(layers.Layer):
    """
    Convert sequence of tokens into sequence of vector
    """
    def __init__(self, seq_len, vocab_size, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = self.get_positional_encoding(seq_len, embedding_dim)

    def get_positional_encoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in range(int(d / 2)):
                denominator = pow(n, 2 * i / d)
                P[k, 2 * i] = np.sin(k / denominator)
                P[k, 2 * i + 1] = np.cos(k / denominator)
        
        return P
    
    def call(self, x):
        """
        `x`: list of token (batch_size, seq_len)
        - output with shape: (batch_size, seq_len, embedding_dim)
        """
        x = self.embedding(x)
        x += self.positional_encoding

        return x