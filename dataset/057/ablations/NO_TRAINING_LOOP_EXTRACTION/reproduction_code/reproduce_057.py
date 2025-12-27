import os
import pathlib
import random
import string
import re
import numpy as np
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings
import keras
from keras import layers

os.environ['KERAS_BACKEND'] = 'tensorflow'

def create_random_text_data(num_samples, max_length):
    return [''.join(random.choices(string.ascii_lowercase, k=random.randint(1, max_length))) for _ in range(num_samples)]

source_sentences = create_random_text_data(1000, 40)
target_sentences = create_random_text_data(1000, 40)

text_vectorization = layers.TextVectorization(max_tokens=20000, output_sequence_length=40)
text_vectorization.adapt(source_sentences)

model = keras.Sequential([
    layers.Input(shape=(40,)),
    layers.Embedding(input_dim=20000, output_dim=256),
    layers.LSTM(128),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

try:
    from keras import ops
except ImportError as e:
    print(e)

print(keras.__version__)