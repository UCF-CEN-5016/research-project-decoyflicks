import tensorflow as tf
from object_detection.utils import ops as utils_ops

def reframe_box_masks_to_image_masks(box_masks, crop_size):
    # Reshape box_masks to [num_boxes, height, width, 1] and resize to crop_size
    reshaped_masks = tf.expand_dims(box_masks, axis=3)
    reshaped_masks = tf.image.resize(reshaped_masks, crop_size)
    return tf.squeeze(reshaped_masks, axis=3)

# Simulate dummy data for box masks and boxes
box_masks = tf.constant([[[1.0, 0.0], [0.0, 1.0]]])  # Shape: [num_boxes, height, width]
crop_size = (100, 100)  # Output size of the reframe

# Simulate the function call that causes the error
try:
    detection_masks_reframed = reframe_box_masks_to_image_masks(box_masks, crop_size)
except TypeError as e:
    print("Caught TypeError:", e)

detection_masks_reframed = reframe_box_masks_to_image_masks(box_masks, crop_size)