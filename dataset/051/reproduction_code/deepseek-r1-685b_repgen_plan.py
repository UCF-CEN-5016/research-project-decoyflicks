import numpy as np
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

# This will raise the TypeError
try:
    output = sampler.generate_tokens(
        next_fn=mock_next,
        prompt=prompt,
        max_length=20
    )
except TypeError as e:
    print(f"Error: {e}")
    print("The sampler doesn't accept 'end_token_id' parameter in this Keras version")