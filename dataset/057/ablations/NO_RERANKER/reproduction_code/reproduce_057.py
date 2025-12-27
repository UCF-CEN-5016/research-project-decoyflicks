import os
import numpy as np
import tensorflow as tf
from keras import layers
from keras import Model

os.environ['KERAS_BACKEND'] = 'tensorflow'

def create_sample_dataset(vocab_size=10000, num_sentences=1000, sentence_length=10):
    vocab = ['word' + str(i) for i in range(vocab_size)]
    source_sentences = [' '.join(np.random.choice(vocab, sentence_length)) for _ in range(num_sentences)]
    target_sentences = [' '.join(np.random.choice(vocab, sentence_length)) for _ in range(num_sentences)]
    return source_sentences, target_sentences

source_data, target_data = create_sample_dataset()

source_vectorizer = layers.TextVectorization(max_tokens=10000, output_sequence_length=10)
target_vectorizer = layers.TextVectorization(max_tokens=10000, output_sequence_length=10)

source_vectorizer.adapt(source_data)
target_vectorizer.adapt(target_data)

# Attempt to import keras.ops
try:
    from keras import ops
except ImportError as e:
    print(e)