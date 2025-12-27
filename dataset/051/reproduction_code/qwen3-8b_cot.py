import tensorflow as tf
from keras_nlp import samplers

# Dummy inputs to simulate the call to the sampler
next_token = tf.constant([1])
prompt = tf.constant([[1, 2, 3]])

# Initialize the GreedySampler without end_token_id
sampler = samplers.GreedySampler()

# Incorrect usage: passing end_token_id as a keyword argument to the __call__ method
generated_tokens = sampler(next_token, prompt, end_token_id=123)

print(generated_tokens)

sampler = samplers.GreedySampler(end_token_id=123)
generated_tokens = sampler(next_token, prompt)

import tensorflow as tf
from keras_nlp import samplers

# Dummy inputs to simulate the call to the sampler
next_token = tf.constant([1])
prompt = tf.constant([[1, 2, 3]])

# Initialize the GreedySampler with end_token_id
sampler = samplers.GreedySampler(end_token_id=123)

# Call the sampler with the next_token and prompt
generated_tokens = sampler(next_token, prompt)

print(generated_tokens)