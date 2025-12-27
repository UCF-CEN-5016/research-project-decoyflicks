import keras_nlp
import numpy as np

# Generate a dummy prompt (shape: (batch_size, sequence_length))
prompt = np.random.random((1, 10))

# Initialize the GreedySampler
sampler = keras_nlp.samplers.GreedySampler()

# Call the `__call__` method with an invalid keyword argument `end_token_id`
generated_tokens = sampler(prompt, end_token_id=123)

import keras_nlp
import numpy as np

prompt = np.random.random((1, 10))
sampler = keras_nlp.samplers.GreedySampler()
generated_tokens = sampler(prompt)