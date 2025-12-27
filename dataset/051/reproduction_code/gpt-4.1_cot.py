import keras_nlp

# Dummy function just to satisfy the sampler's input requirements.
def next_fn(tokens):
    # This function should return logits for the next token.
    # For minimal reproduction, return dummy logits.
    import tensorflow as tf
    batch_size = tf.shape(tokens)[0]
    vocab_size = 10
    return tf.random.uniform((batch_size, vocab_size))

# Initialize the sampler
sampler = keras_nlp.samplers.GreedySampler()

# Dummy prompt tokens (batch_size=1, sequence_length=5)
import tensorflow as tf
prompt = tf.constant([[1, 2, 3, 4, 5]])

# Try calling sampler with end_token_id (which triggers the error)
generated_tokens = sampler(
    next_fn,
    prompt,
    end_token_id=1,  # This will cause TypeError in the reported version
    max_length=10
)