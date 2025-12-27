import keras
import keras_nlp
import numpy as np
import random

np.random.seed(42)

test_eng_texts = ['Hello', 'How are you?', 'What is your name?', 'I love programming.']

def decode_sequences(input_sentences):
    batch_size = 1
    start = np.array([[1]])
    pad = np.zeros((1, 10))
    prompt = np.concatenate((start, pad), axis=-1)

    next = lambda x: np.array([[2]])
    
    generated_tokens = keras_nlp.samplers.GreedySampler()(
        next,
        prompt,
        stop_token_ids=[2],  # This will trigger the TypeError
        index=1,
    )
    return generated_tokens

translated = decode_sequences([random.choice(test_eng_texts)])