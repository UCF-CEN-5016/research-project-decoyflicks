import tensorflow as tf
from keras_nlp.samplers import GreedySampler

batch_size = 32
sequence_length = 50

input_data = tf.random.uniform((batch_size, sequence_length), maxval=1000, dtype=tf.int32)

tokenizer = tf.keras.layers.StringLookup(vocabulary=list('abcdefghijklmnopqrstuvwxyz'), mask_token='<pad>')
encoded_sequences = tokenizer(input_data)

greedy_sampler = GreedySampler()

def decode_sequences(encoded_seq):
    return greedy_sampler(encoded_seq)

try:
    decoded_tokens = decode_sequences(encoded_sequences)
except TypeError as e:
    print(e)