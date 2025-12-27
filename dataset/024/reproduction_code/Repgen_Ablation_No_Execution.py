import tensorflow as tf

batch_size = 2
image_size = 321
num_classes = 1000

input_data = tf.random.uniform((batch_size, image_size, image_size, 3), dtype=tf.float32)

from delf.python.training.model import Delg

delg_instance = Delg(block3_strides=True, use_dim_reduction=True)
delg_instance.init_classifiers(num_classes=num_classes)

# This should raise ModuleNotFoundError
output = delg_instance.forward_pass(input_data)