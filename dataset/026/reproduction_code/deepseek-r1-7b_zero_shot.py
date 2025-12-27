import tensorflow as tf
from object_detection.utils import ops

# Simulated code to mimic the notebook's execution
detection_boxes = tf.random.uniform((10, 4), dtype=tf.float32)
image_height, image_width = 500, 600
detection_masks = tf.random.uniform((10, 28, 28, 2), dtype=tf.float32)

# Fix: Include 'box_ind' parameter in the function call
detection_masks_reframed = ops.reframe_box_masks_to_image_masks(
    detection_masks,
    detection_boxes,
    box_ind=tf.range(10),
    crop_size=[image_height, image_width],
    extrapolation_value=0.0
)