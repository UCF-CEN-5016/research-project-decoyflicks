import tensorflow as tf

# Simulate a scenario where the input list is empty (e.g., no predicted boxes)
values = []

# Attempt to concatenate using ConcatV2 with an empty list
try:
    result = tf.raw_ops.ConcatV2(input=values, num_splits=[1], axis=0)
    print("ConcatV2 executed successfully.")
except ValueError as e:
    print(f"Error occurred: {e}")

import tensorflow as tf

values = []

if len(values) < 2:
    # Handle empty or insufficient input (e.g., return a zero tensor)
    result = tf.zeros(shape=[0, 4], dtype=tf.float32)
else:
    result = tf.raw_ops.ConcatV2(input=values, num_splits=[1], axis=0)

print("Result shape:", result.shape)