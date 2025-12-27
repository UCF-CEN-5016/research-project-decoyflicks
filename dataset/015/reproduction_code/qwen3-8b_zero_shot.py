

import tensorflow as tf

def show_batch(data, num_of_examples):
    tf.print("Data:", data)

show_batch([1, 2, 3], num_of_examples=5)