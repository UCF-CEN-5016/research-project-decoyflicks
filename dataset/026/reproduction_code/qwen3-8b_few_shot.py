import tensorflow as tf
import numpy as np
from object_detection.utils import ops as utils_ops

# Simulate detection outputs (common in object detection pipelines)
detection_boxes = np.random.rand(10, 4)  # 10 boxes, 4 coordinates each
detection_masks = np.random.rand(10, 28, 28)  # 10 masks, 28x28
image_np = np.random.rand(28, 28)  # Simulated image

# This call will trigger the error
try:
    # Old API expects 'box_ind' parameter (no longer supported in newer versions)
    utils_ops.reframe_box_masks_to_image_masks(
        detection_masks, detection_boxes,
        image_np.shape[1], image_np.shape[2],
        box_ind=tf.range(10)  # Invalid argument in newer versions
    )
except TypeError as e:
    print("Error:", e)