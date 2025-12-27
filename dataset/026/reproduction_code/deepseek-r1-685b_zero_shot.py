import tensorflow as tf
import numpy as np
from object_detection.utils import ops

# Create dummy inputs
num_boxes = 2
image_height = 100
image_width = 100
box_masks = tf.random.uniform([num_boxes, 10, 10], dtype=tf.float32)
detection_boxes = tf.constant([[0.1, 0.1, 0.5, 0.5], [0.3, 0.3, 0.8, 0.8]], dtype=tf.float32)

# Call the problematic function
try:
    masks = ops.reframe_box_masks_to_image_masks(
        box_masks, detection_boxes, 
        image_height, image_width
    )
    print(masks.shape)
except TypeError as e:
    print(f"Error: {e}")