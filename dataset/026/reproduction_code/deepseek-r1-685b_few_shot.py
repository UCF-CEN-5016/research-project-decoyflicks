import tensorflow as tf
from object_detection.utils import ops
import numpy as np

# Simulate detection outputs
num_boxes = 5
image_height, image_width = 300, 400
detection_masks = tf.random.uniform((num_boxes, 28, 28))  # Example mask
detection_boxes = tf.random.uniform((num_boxes, 4))  # Example boxes

# This will raise TypeError in TF 2.10+
try:
    masks_reframed = ops.reframe_box_masks_to_image_masks(
        detection_masks, 
        detection_boxes,
        image_height,
        image_width
    )
    print("Successfully reframed masks")
except TypeError as e:
    print(f"Error: {e}")
    print("This occurs because 'box_ind' argument was removed in newer TF versions")