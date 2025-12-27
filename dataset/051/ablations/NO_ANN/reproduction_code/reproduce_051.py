import keras
import keras_nlp
import numpy as np
import random
import tensorflow as tf

np.random.seed(42)
random.seed(42)

vocab_size = 10000
max_length = 20
embedding_dim = 256
input_sentence = 'This is a test sentence.'
input_data = np.random.randint(0, vocab_size, size=(1, max_length))

start_token = np.array([[1]])
pad_token = np.array([[0]])
prompt = np.concatenate((start_token, pad_token), axis=-1)

def next(prompt, cache, index):
    return np.random.rand(1, vocab_size), None, cache

sampler = keras_nlp.samplers.GreedySampler()

try:
    generated_tokens = sampler(next, prompt, end_token_id=2)
except TypeError as e:
    print(e)