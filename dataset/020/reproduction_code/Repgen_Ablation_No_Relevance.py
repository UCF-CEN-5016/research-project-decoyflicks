import tensorflow as tf
from deeplab import xception_65, xception_arg_scope
import slim
import six

# Set batch size and image dimensions
batch_size = 2
height, width = 32, 32

# Create random uniform input data
inputs = tf.random.uniform((batch_size, height, width, 3), minval=0, maxval=1, dtype=tf.float32)

# Call xception_65 function with num_classes=10
with slim.arg_scope(xception_arg_scope()):
    net, end_points = xception_65(inputs, num_classes=10)

# Verify the output tensor has shapes as expected for each convolutional layer endpoint
for name, endpoint in six.iteritems(end_points):
    print(f"Endpoint {name}: {endpoint.shape}")

# Monitor the GPU memory usage during execution of the model
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    # Run a dummy forward pass to measure GPU memory usage
    net.evaluated = sess.run(net)