import tensorflow as tf
from deeplab import xception

batch_size = 2
height = width = 65
inputs = tf.random.uniform((batch_size, height, width, 3), minval=0, maxval=1, dtype=tf.float32)

output = xception._xception_small(inputs, num_classes=None, global_pool=False, output_stride=16)
print(output)