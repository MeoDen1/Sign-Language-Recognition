import numpy as np
import tensorflow as tf
# # print(np.random.randint(1, 9, 20))

# decoder_input = np.random.randint(1, 9, 20)
# decoder_input = np.array([decoder_input])

# print(decoder_input.shape[-1])

from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

# test_str = ["<sos> hello world", "hello test"]
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(test_str)
# print(tokenizer.word_index)
# print(pad_sequences(tokenizer.texts_to_sequences(test_str), maxlen=4, padding='post'))

# print(tf.reduce_mean([[1.0], [2]]))


t0 = tf.convert_to_tensor(np.zeros((1, 2, 3, 4)))
print(t0)
t1 = tf.convert_to_tensor(np.ones((1, 1, 3)))

print(t0 + t1)