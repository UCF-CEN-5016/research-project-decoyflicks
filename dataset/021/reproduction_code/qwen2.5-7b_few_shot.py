import tensorflow as tf

def concatenate_boxes(boxes):
    if not boxes:
        boxes = [tf.zeros([0, 4])]  # default tensor for empty boxes
    return tf.concat(boxes, axis=1)

# Simulate an empty list of tensors (no boxes)
boxes = []

# Now concatenate safely
result = concatenate_boxes(boxes)
print(result)