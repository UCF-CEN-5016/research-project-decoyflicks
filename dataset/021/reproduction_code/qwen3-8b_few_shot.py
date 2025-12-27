import tensorflow as tf

# Simulate an empty list of tensors (no boxes)
boxes = []

# Attempt to concatenate the empty list
result = tf.concat(boxes, axis=1)

# This will raise the error
print(result)

import tensorflow as tf

# Simulate an empty list of tensors (no boxes)
boxes = []

# Ensure at least one tensor (even if empty) is present
if not boxes:
    boxes = [tf.zeros([0, 4])]  # default tensor for empty boxes

# Now concatenate safely
result = tf.concat(boxes, axis=1)
print(result)