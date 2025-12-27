import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
from tensorflow.keras import utils
from tensorflow.keras.nlp import samplers
from tensorflow.keras.nlp import layers as nlp_layers
import tensorflow_datasets as tfds

# Load dataset
dataset, metadata = tfds.load('wmt14_translate/en-fr',
                              with_info=True,
                              as_supervised=True)

train_examples, val_examples = dataset['train'], dataset['validation']

def encode(lang1, lang2):
    lang1 = [vocab_lang1.size + 2] + vocab_lang1.encode(
        lang1.numpy()) + [vocab_lang1.size + 3]

    lang2 = [vocab_lang2.size + 2] + vocab_lang2.encode(
        lang2.numpy()) + [vocab_lang2.size + 3]

    return lang1, lang2

def tf_encode(en, fr):
    result_en, result_fr = tf.py_function(encode, [en, fr], [tf.int64, tf.int64])
    result_en.set_shape([None])
    result_fr.set_shape([None])

    return result_en, result_fr

vocab_size = 15000
vocab_lang1 = preprocessing.StringLookup(vocabulary=utils.compute_vocabulary(train_examples.map(lambda x, y: x).batch(32), vocab_size))
vocab_lang2 = preprocessing.StringLookup(vocabulary=utils.compute_vocabulary(train_examples.map(lambda x, y: y).batch(32), vocab_size))

def encode_map(examples):
    en, fr = examples
    en = vocab_lang1([en])
    fr = vocab_lang2([fr])
    en, fr = tf_encode(en[0], fr[0])
    return en, fr

def filter_max_length(x, y, max_length=40):
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.size(y) <= max_length)

train_dataset = train_examples.map(encode_map).filter(filter_max_length).padded_batch(
    batch_size=64, padded_shapes=(100, 100))

val_dataset = val_examples.map(encode_map).filter(filter_max_length).padded_batch(
    batch_size=64, padded_shapes=(100, 100))

model = keras.Sequential([
    nlp_layers.TransformerEncoder(
        num_layers=2,
        num_heads=2,
        mlp_dim=32,
        dropout=0.1,
        layer_norm_epsilon=1e-6,
        output_layer=None,
    ),
    nlp_layers.TransformerDecoder(
        num_layers=2,
        num_heads=2,
        mlp_dim=32,
        dropout=0.1,
        layer_norm_epsilon=1e-6,
    )
])

next_token = layers.Dense(vocab_size)

def decode_sequences(input_sentences):
    start = tf.constant([vocab_lang2.size + 2])
    pad = tf.constant([0] * (50 - 1))
    prompt = tf.concat((start, pad), axis=-1)
    generated_tokens = samplers.GreedySampler()(
        next_token,
        prompt,
        end_token_id=vocab_lang2.size + 3,
    )
    return generated_tokens

decode_sequences(tf.constant([vocab_lang2.size + 2] * 50))