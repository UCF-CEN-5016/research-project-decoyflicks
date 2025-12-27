import tensorflow as tf
from object_detection.utils import ops as utils_ops

# Simulate dummy data for box masks and boxes
box_masks = tf.constant([[[1.0, 0.0], [0.0, 1.0]]])  # Shape: [num_boxes, height, width]
box_ind = tf.constant([0])  # Index of the box (should be positional in the function)
crop_size = (100, 100)  # Output size of the reframe

# Simulate the function call that causes the error
# The function 'reframe_box_masks_to_image_masks' is called with 'box_ind' as a keyword argument
# But the function does not accept this argument, hence the error
try:
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
        box_masks, box_ind, crop_size
    )
except TypeError as e:
    print("Caught TypeError:", e)

detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
    box_masks, crop_size
)