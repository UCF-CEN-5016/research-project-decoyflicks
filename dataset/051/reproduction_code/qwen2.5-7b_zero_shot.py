import keras_nlp
import numpy as np

# Generate a dummy prompt (shape: (batch_size, sequence_length))
prompt = np.random.random((1, 10))

# Initialize the GreedySampler
sampler = keras_nlp.samplers.GreedySampler()

# Call the sampler with the prompt
generated_tokens = sampler(prompt)