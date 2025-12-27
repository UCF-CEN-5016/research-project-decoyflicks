import keras
import keras_nlp
import random
import tensorflow as tf

tf.random.set_seed(42)

vocab_size = 10000
embedding_dim = 256
hidden_units = 512
max_length = 20

input_sentence = 'Hello, how are you?'
test_eng_texts = ['Hello, how are you?', 'What is your name?', 'Where are you from?']

def decode_sequences(input_sentences):
    start = tf.constant([[1]])
    pad = tf.constant([[0]])
    prompt = tf.concat((start, pad), axis=-1)
    sampler = keras_nlp.samplers.GreedySampler()
    generated_tokens = sampler(prompt, end_token_id=2)  # This will raise the TypeError
    return generated_tokens

try:
    translated = decode_sequences([input_sentence])
except TypeError as e:
    assert "unexpected keyword argument 'end_token_id'" in str(e)
    print('Caught expected TypeError:', e)