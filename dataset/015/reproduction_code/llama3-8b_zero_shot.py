import tensorflow as tf

def show_batch(batch):
    pass

batch = tf.data.Dataset.from_tensors([1, 2]).batch(1)
show_batch(batch)