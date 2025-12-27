import tensorflow as tf

# Simulating the scenario where the input list is empty
boxes = []

# Check if the boxes list is non-empty before concatenating
if boxes:
    concat_boxes = tf.concat(boxes, axis=1)
    print(concat_boxes)
else:
    print("The boxes list is empty.")