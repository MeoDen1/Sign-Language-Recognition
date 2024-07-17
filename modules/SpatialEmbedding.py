import tensorflow as tf
import numpy as np
from keras import layers

class SpatialEmbedding(layers.Layer):
    """
    Convert sequence of frames into sequence of vectors
    """
    def __init__(self, seq_len, embedding_dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len

        self.conv_2d_1 = layers.Conv2D(16, 3, activation='relu')
        self.max_pooling_1 = layers.MaxPooling2D()
        self.conv_2d_2 = layers.Conv2D(32, 3, activation='relu')
        self.W_1 = layers.Dense(embedding_dim)
        self.flatten = layers.Flatten()

        # position encoding
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
        `x`: shape (batch_size, seq_len, height, width, channels)
        - output with shape (batch_size, seq_len, embedding_dim)
        """
        # x: (batch_size, seq_len, height, width, channels)
        processed_frames = []

        for i in range(self.seq_len):
            frame = x[:, i, :, :, :]
            x1 = self.conv_2d_1(frame)
            x1 = self.max_pooling_1(x1)
            x1 = self.conv_2d_2(x1)
            x1 = self.flatten(x1)
            x1 = self.W_1(x1)
            processed_frames.append(x1)

        output = tf.stack(processed_frames, axis=1)
        output += self.positional_encoding

        return output