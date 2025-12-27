import keras_nlp
import tensorflow as tf

# Dummy next token logits function
def next_token_logits_fn(tokens):
    batch_size = tf.shape(tokens)[0]
    vocab_size = 10
    # Return fixed logits for simplicity
    return tf.ones((batch_size, vocab_size))

# Dummy input prompt tokens
prompt = tf.constant([[1, 2, 3]])

# Instantiate sampler
sampler = keras_nlp.samplers.GreedySampler()

# Attempt to call with unsupported argument 'end_token_id'
# This will raise: TypeError: Sampler.__call__() got an unexpected keyword argument 'end_token_id'
generated = sampler(
    next_token_logits_fn,
    prompt,
    end_token_id=0,  # Unsupported argument triggers the error
)

print(generated)