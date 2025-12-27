import tensorflow as tf

# Simulating the scenario where the input list is empty
boxes = []

# This line will trigger the error if the list is empty
concat_boxes = tf.concat(boxes, axis=1)

# This will never be reached due to the error
print(concat_boxes)

import tensorflow as tf

boxes = []
tf.concat(boxes, axis=1)