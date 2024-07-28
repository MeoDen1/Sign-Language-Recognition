import tensorflow as tf
import numpy as np
from keras import layers, models

from modules.WordEmbedding import WordEmbedding
from modules.MultiHeadAttention import MultiHeadAttention
from modules.ModelLayer import *

class DecoderLayer(layers.Layer):
    def __init__(self, heads, d_k, d_v, d_ff, d_model, rate, **kwargs):
        super().__init__(**kwargs)
        # Masked multi head attention
        self.multi_head_attention_1 = MultiHeadAttention(heads, d_k, d_v, d_model)
        self.dropout_1 = layers.Dropout(rate)
        self.add_norm_1 = AddNormalization()

        self.multi_head_attention_2 = MultiHeadAttention(heads, d_k, d_v, d_model)
        self.dropout_2 = layers.Dropout(rate)
        self.add_norm_2 = AddNormalization()

        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout_3 = layers.Dropout(rate)
        self.add_norm_3 = AddNormalization()

    def call(self, x, encoder_output, lookahead_mask, padding_mask, training):
        multihead_output_1 = self.multi_head_attention_1(x, x, x, mask=lookahead_mask)
        # Output shape: (batch_size, seq_len, d_model)
        multihead_output_1 = self.dropout_1(multihead_output_1, training=training)
        multihead_output_1 = self.add_norm_1(multihead_output_1, x)
        # Output shape: (batch_size, seq_len, d_model)

        multihead_output_2 = self.multi_head_attention_2(multihead_output_1, encoder_output, encoder_output, mask=padding_mask)
        # Output shape: (batch_size, seq_len, d_model)
        multihead_output_2 = self.dropout_2(multihead_output_2)
        multihead_output_2 = self.add_norm_2(multihead_output_2, multihead_output_1)
        # Output shape: (batch_size, seq_len, d_model)

        feed_forward_output = self.feed_forward(multihead_output_2)
        feed_forward_output = self.dropout_3(feed_forward_output)
        feed_forward_output = self.add_norm_3(feed_forward_output, multihead_output_2)
        # Output shape: (batch_size, seq_len, d_model)

        return feed_forward_output
    

class Decoder(layers.Layer):
    def __init__(self, vocab_size, seq_len, heads, d_k, d_v, d_ff, d_model, rate, N=6, **kwargs):
        super().__init__(**kwargs)
        self.word_embedding = WordEmbedding(seq_len, vocab_size, d_model)
        self.decoder_layers = [DecoderLayer(heads, d_k, d_v, d_ff, d_model, rate) for _ in range(N)]

    def call(self, output_target, encoder_output, lookahead_mask, padding_mask, training):
        x = self.word_embedding(output_target)

        for i, layer in enumerate(self.decoder_layers):
            x = layer(x, encoder_output, lookahead_mask=lookahead_mask, padding_mask=padding_mask, training=training)

        return x