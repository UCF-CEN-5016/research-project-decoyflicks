import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import Xception
import matplotlib.pyplot as plt

# Set parameters
batch_size = 2
height, width = 65, 65
num_classes = 10

# Create random input data
input_data = tf.random.uniform((batch_size, height, width, 3), dtype=tf.float32)

# Call xception function
model = Xception(weights=None, input_shape=(height, width, 3), include_top=False)
output = model(input_data, training=True)

# Assert that the output tensor contains NaN values in the logits calculation
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(output)
    assert np.isnan(result).any(), "The output tensor does not contain NaN values"

# Verify that the computation graph produces intermediate tensors with non-finite values
graph = tf.get_default_graph()
logits_tensor = None
for op in graph.get_operations():
    if 'global_pool' in op.name:
        logits_tensor = op.outputs[0]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(logits_tensor)
    assert not np.isfinite(result).all(), "The computation graph does not produce intermediate tensors with non-finite values"

# Monitor GPU memory usage during execution
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
info = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(f"Initial GPU Memory: {info.used} bytes")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(output)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"Final GPU Memory: {info.used} bytes")

# Assert that GPU memory exceeds expected threshold for a model of this complexity
expected_threshold = 1e9  # Example threshold, adjust as needed
assert info.used > expected_threshold, "GPU memory usage does not exceed expected threshold"

# Ensure plots are displayed
plt.show()