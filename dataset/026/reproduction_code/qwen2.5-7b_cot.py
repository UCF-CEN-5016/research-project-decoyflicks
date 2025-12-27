import tensorflow as tf
import numpy as np

# Simulate the outputs of a detection model
image_np = np.random.rand(480, 640, 3).astype(np.uint8)  # Simulated input image
detection_boxes = np.random.rand(10, 4).astype(np.float32)  # Simulated bounding boxes
detection_masks = np.random.rand(10, 8, 8).astype(np.float32)  # Simulated masks
num_boxes = 10

def reframe_box_masks_to_image_masks_default(box_masks, detection_boxes, image_height, image_width):
    # Reframe box masks to image masks
    box_ind = tf.range(num_boxes)
    return tf.image.crop_and_resize(box_masks, detection_boxes, box_ind, [image_height, image_width])

# Simulate the function call that triggers the error
reframe_box_masks_to_image_masks_default(detection_masks, detection_boxes, 480, 640)

def reframe_box_masks_to_image_masks_updated(box_masks, detection_boxes, image_height, image_width):
    # Reframe box masks to image masks without 'box_ind'
    return tf.image.crop_and_resize(box_masks, detection_boxes, [image_height, image_width])

# Corrected function call
reframe_box_masks_to_image_masks_updated(detection_masks, detection_boxes, 480, 640)