import tensorflow as tf
import numpy as np
from keras import layers, models

from SpatialEmbedding import SpatialEmbedding
from MultiHeadAttention import MultiHeadAttention

class Encoder(layers.Layer):
    def __init__(self, heads, seq_len, d_k, d_v, d_model):
        self.spatial_embedding = SpatialEmbedding(seq_len, d_model)
        self.multi_head_attention = MultiHeadAttention(heads, d_k, d_v, d_model)
        