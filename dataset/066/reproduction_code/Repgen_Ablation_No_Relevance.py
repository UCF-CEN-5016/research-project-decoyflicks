import tensorflow as tf
from text_generation_test import token_logits_fn

# Define batch size
batch_size = 64

# Create random uniform input data
max_sequence_length = 100
input_data = tf.random.uniform((batch_size, max_sequence_length), minval=0, maxval=10, dtype=tf.int32)

# Call token_logits_fn with prompt_tokens as input
prompt_tokens = tf.convert_to_tensor([[0]])
output_tokens = token_logits_fn(prompt_tokens)

# Verify output contains NaN values in generated text sequence
contains_nan = tf.reduce_any(tf.math.is_nan(output_tokens))
print("Output contains NaN values:", contains_nan.numpy())