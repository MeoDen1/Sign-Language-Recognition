import tensorflow as tf
import numpy as np
from keras import layers, models

from modules.SpatialEmbedding import SpatialEmbedding
from modules.MultiHeadAttention import MultiHeadAttention
from modules.ModelLayer import *

class EncoderLayer(layers.Layer):
    def __init__(self, heads, d_k, d_v, d_ff, d_model, rate, **kwargs):
        super().__init__(**kwargs)
        self.multi_head_attention = MultiHeadAttention(heads, d_k, d_v, d_model)
        self.dropout_1 = layers.Dropout(rate)
        self.add_norm_1 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout_2 = layers.Dropout(rate)
        self.add_norm_2 = AddNormalization()

    def call(self, x, padding_mask, training):
        multihead_output = self.multi_head_attention(x, x, x, mask=padding_mask)
        # Output shape: (batch_size, seq_len, d_model)
        multihead_output = self.dropout_1(multihead_output, training=training)
        multihead_output = self.add_norm_1(x, multihead_output)

        feed_forward_output = self.feed_forward(multihead_output)
        feed_forward_output = self.dropout_2(feed_forward_output, training=training)
        # Output shape: (batch_size, seq_len, d_model)
        feed_forward_output = self.add_norm_2(multihead_output, feed_forward_output)

        return feed_forward_output

# Possible error:
# Only input tensors may be passed as positional arguments.
# The following argument value should be passed as a keyword argument
class Encoder(layers.Layer):
    def __init__(self, seq_len, heads, d_k, d_v, d_ff, d_model, rate, N=6, **kwargs):
        super().__init__(**kwargs)
        self.spatial_embedding = SpatialEmbedding(seq_len, d_model)
        self.encoder_layers = [EncoderLayer(heads, d_k, d_v, d_ff, d_model, rate) for _ in range(N)]

    def call(self, x, padding_mask, training):
        x = self.spatial_embedding(x)

        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, padding_mask=padding_mask, training=training)

        return x