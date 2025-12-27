import random
import numpy as np

# Minimal setup for Keras and Keras NLP
from tensorflow.keras import backend as K
from keras_nlp.samplers import GreedySampler

# Triggering condition: Call `decode_sequences()` with unexpected keyword argument `'end_token_id'`
def decode_sequences(input_sentences):
    prompt = ...  # irrelevant for this bug
    generated_tokens = GreedySampler()(
        next,  # should not be called here
        prompt,  # irrelevant for this bug
        end_token_id=12345  # unexpected keyword argument
    )
    return generated_tokens

# Trigger the bug
input_sentence = random.choice(["Hello", "World"])
translated = decode_sequences([input_sentence])
print(translated)

TypeError: Sampler.__call__() got an unexpected keyword argument 'end_token_id'