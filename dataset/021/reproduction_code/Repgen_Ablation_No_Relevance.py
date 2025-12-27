import tensorflow as tf
from deeplab import _xception_small

batch_size = 2
height = 32
width = 32
global_pool = True

# Create random uniform input data with shape (batch_size, height, width, 3)
input_data = tf.random.uniform((batch_size, height, width, 3), minval=0, maxval=1, dtype=tf.float32)

# Call the _xception_small function
output, end_points = _xception_small(input_data, num_classes=None, is_training=False, reuse=True, output_stride=8, global_pool=False)

# Verify that the output contains NaN values in the logits tensor
assert tf.reduce_any(tf.math.is_nan(end_points['predictions'])), "Logits tensor should contain NaN values"

# Monitor the memory usage of the GPU during execution
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Assert that the GPU memory usage exceeds a predefined threshold indicating an out-of-memory error
with tf.device('/GPU:0'):
    output.eval(session=tf.compat.v1.Session())