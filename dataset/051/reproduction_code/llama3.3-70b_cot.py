import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_nlp
import random

# Simulate the decode_sequences function
def decode_sequences(input_sentence):
    # Creating a dummy model and next function for demonstration
    model = keras.Sequential([layers.Dense(10)])
    def next(prompt):
        return model(prompt)
    
    # Simulating the start and pad tokens
    start = np.array([1])
    pad = np.array([0] * 10)
    
    prompt = np.concatenate((start, pad), axis=-1)
    
    # Attempting to use the GreedySampler with end_token_id
    try:
        generated_tokens = keras_nlp.samplers.GreedySampler()(next, prompt, end_token_id=2)
    except TypeError as e:
        print(f"Error: {e}")

# Triggering the bug
decode_sequences("Hello")