import tensorflow as tf

def show_batch(data):
    tf.print("Data:", data)

num_of_examples = 5
show_batch([1, 2, 3])