import tensorflow as tf
from official.nn_blocks import MultiHeadSelfAttentionBlock

# Define test parameters
batch_size = 4
in_filters = 64
input_size = 128
num_heads = 7
key_dim = 8
value_dim = 64
query_h_strides = 1
query_w_strides = 1
kv_strides = 1

# Create input tensor
input_tensor = tf.random.uniform((batch_size, input_size, input_size, in_filters), dtype=tf.float32)

# Initialize MultiHeadSelfAttentionBlock
multi_head_attention_block = MultiHeadSelfAttentionBlock(
    num_heads=num_heads,
    key_dim=key_dim,
    value_dim=value_dim,
    query_h_strides=query_h_strides,
    query_w_strides=query_w_strides,
    kv_strides=kv_strides,
    name='mh_self_attn'
)

# Pass input tensor to the MultiHeadSelfAttentionBlock
output_tensor = multi_head_attention_block(input_tensor, training=True)

# Verify output shape
assert output_tensor.shape == (batch_size, input_size, input_size, in_filters), "Output tensor has incorrect shape"

# Check for NaN or Inf values
if tf.math.is_nan(output_tensor).any().numpy():
    print("NaN values found in output tensor")
if tf.math.is_inf(output_tensor).any().numpy():
    print("Inf values found in output tensor")

# Print maximum value in the output tensor
print("Maximum value in output tensor:", tf.reduce_max(output_tensor).numpy())

# Monitor memory usage and store it as a variable
# (Memory usage monitoring is typically done using system-level tools or libraries, not directly in Python code)