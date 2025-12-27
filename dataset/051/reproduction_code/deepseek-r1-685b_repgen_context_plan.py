import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from keras_nlp.samplers import GreedySampler

# Minimal setup to reproduce the sampler error
vocab_size = 100
seq_length = 10

# Mock model that returns logits
def mock_next(prompt, state=None):
    batch_size = prompt.shape[0]
    return np.random.rand(batch_size, seq_length, vocab_size), state

# Create sampler and prompt
sampler = GreedySampler()
prompt = np.array([[1, 2, 3]])  # Sample input tokens

# Define a sampling function without 'end_token_id' parameter
def sample_fn(next_func, prompt, max_length):
    state = None
    result = []
    for _ in range(max_length):
        logits, state = next_func(prompt, state)
        next_token = np.argmax(logits[:, -1], axis=-1)
        result.append(next_token)
        prompt = np.concatenate([prompt, np.expand_dims(next_token, axis=1)], axis=1)
    return np.array(result)

# Call the sampling function
output = sample_fn(mock_next, prompt, max_length=20)
print(output)