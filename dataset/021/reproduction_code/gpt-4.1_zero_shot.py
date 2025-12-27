import tensorflow as tf

empty_tensor = tf.constant([], shape=(0, 4), dtype=tf.float32)
tf.concat([empty_tensor], axis=0)