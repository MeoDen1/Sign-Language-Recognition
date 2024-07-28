import tensorflow as tf
import numpy as np
from keras import layers

import sys, os
sys.path.append(os.path.abspath('.'))


from dataset import Dataset, pre_processing_1
from modules.SpatialEmbedding import SpatialEmbedding
from modules.MultiHeadAttention import MultiHeadAttention
from modules.Encoder import Encoder
from modules.Decoder import Decoder
from Model import TransformerModel

heads = 4
d_k = 64
d_v = 64
d_ff = 64
encoder_seq_len = 200
decoder_seq_len = 20
d_model = 128
rate=0.2
vocab_size=10

# TEST 1: Dataset loader
print("\nTEST 1: Load Dataset")
dataset = Dataset()
sample_data = dataset.load(size=1)
tokenizer, sample_encoder_input, sample_decoder_input, _ = pre_processing_1(sample_data, encoder_seq_len, decoder_seq_len)

vocab_size = len(tokenizer.word_index)
print(sample_encoder_input.shape)
print(sample_decoder_input.shape)
print(vocab_size)

# TEST 2: Spatial Embedding
print("\nTEST 2: Spatial Embedding")
spatial_embedding = SpatialEmbedding(encoder_seq_len, d_model)
se_output = spatial_embedding(sample_encoder_input)
print(se_output)


# TEST 3: Multi Head Attention
print("\nTEST 3: Multi Head Attention")
multi_head_attention = MultiHeadAttention(heads, d_k, d_v, d_model)
mha_output = multi_head_attention(se_output, se_output, se_output)
print(mha_output)


# TEST 4: Encoder
print("\nTEST 4: Encoder")
encoder = Encoder(encoder_seq_len, heads, d_k, d_v, d_ff, d_model, rate)
encoder_output = encoder(sample_encoder_input, None, training=True)
print(encoder_output)

# TEST 5: Decoder
print("\nTEST 5: Decoder")
decoder = Decoder(vocab_size, decoder_seq_len, heads, d_k, d_v, d_ff, d_model, rate)
decoder_output = decoder(sample_decoder_input, encoder_output, None, None, training=True)
print(decoder_output)


# TEST 6: Model
print("\nTEST 6: Model")
model = TransformerModel(encoder_seq_len, vocab_size, decoder_seq_len, heads, d_k, d_v, d_ff, d_model, rate)
model_output = model(sample_encoder_input, sample_decoder_input, training=False)
print(model_output)