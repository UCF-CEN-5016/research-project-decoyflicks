import tensorflow as tf
from object_detection.utils import ops
import numpy as np

def simulate_detection_outputs(num_boxes, image_height, image_width):
    detection_masks = tf.random.uniform((num_boxes, 28, 28))
    detection_boxes = tf.random.uniform((num_boxes, 4))
    return detection_masks, detection_boxes

def reframe_masks_to_image(masks, boxes, image_height, image_width):
    try:
        masks_reframed = ops.reframe_box_masks_to_image_masks(masks, boxes, image_height, image_width)
        print("Successfully reframed masks")
    except TypeError as e:
        print(f"Error: {e}")
        print("This occurs because 'box_ind' argument was removed in newer TF versions")

num_boxes = 5
image_height, image_width = 300, 400
detection_masks, detection_boxes = simulate_detection_outputs(num_boxes, image_height, image_width)
reframe_masks_to_image(detection_masks, detection_boxes, image_height, image_width)