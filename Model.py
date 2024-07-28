import tensorflow as tf
import numpy as np
from keras import layers, models

from modules.Encoder import Encoder
from modules.Decoder import Decoder

class TransformerModel(models.Model):
    def __init__ (self, encoder_seq_len, vocab_size, decoder_seq_len, heads, d_k, d_v, d_ff, d_model, rate, **kwargs):
        super().__init__(**kwargs)

        self.encoder = Encoder(encoder_seq_len, heads, d_k, d_v, d_ff, d_model, rate)
        self.decoder = Decoder(vocab_size, decoder_seq_len, heads, d_k, d_v, d_ff, d_model, rate)

        self.linear = layers.Dense(vocab_size, activation='softmax')

    def encoder_padding_mask(self, input):
        mask = tf.math.equal(input, 0)
        
        # Mask shape: (batch_size, seq_len, height, width, channels)
        mask = tf.reshape(mask, (mask.shape[0], mask.shape[1], -1))
        mask = tf.reduce_all(mask, -1)
        mask = tf.expand_dims(mask, axis=1)
        mask = tf.cast(mask, tf.float32)
        # Output shape (batch_size, 1, seq_len)
        return mask

    def lookahead_mask(self, shape):
        mask = 1 - tf.linalg.band_part(np.ones((shape, shape)), -1, 0)
        return mask
    
    def call(self, encoder_input, decoder_input, training):
        encoder_padding_mask = self.encoder_padding_mask(encoder_input)
        decoder_lookahead_mask = self.lookahead_mask(decoder_input.shape[-1])

        # For encoder, the dot-product shape is (batch_size, heads, encoder_seq_len, -1)
        # -> the mask shape should be (batch_size, 1, encoder_seq_len, 1)
        encoder_output = self.encoder(encoder_input, tf.expand_dims(encoder_padding_mask, axis=-1), training=training)

        # For decoder, the dot-product shape is (batch_size, heads, decoder_seq_len, encoder_seq_len)
        # -> the mask shape should be (batch_size, 1, 1, encoder_seq_len)
        decoder_output = self.decoder(decoder_input, encoder_output, decoder_lookahead_mask, tf.expand_dims(encoder_padding_mask, axis=1), training=training)

        model_output = self.linear(decoder_output)

        return model_output