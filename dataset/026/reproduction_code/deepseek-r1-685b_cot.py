import tensorflow as tf
from object_detection.utils import ops as utils_ops

# Set up minimal test case
def reproduce_bug():
    # Create dummy inputs that match the error scenario
    detection_masks = tf.zeros([1, 28, 28, 1])  # Example mask shape
    detection_boxes = tf.constant([[[0.1, 0.1, 0.5, 0.5]]])  # Example box coordinates
    image_height, image_width = 300, 300  # Example image dimensions
    
    # This is the line that triggers the error
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
        detection_masks, detection_boxes,
        image_height, image_width)
    
    return detection_masks_reframed

try:
    result = reproduce_bug()
    print("Successfully ran without errors")
except TypeError as e:
    print(f"Error reproduced: {e}")