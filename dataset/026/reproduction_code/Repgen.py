import tensorflow as tf
import numpy as np
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils

# Create test data
image_np = np.zeros((300, 400, 3), dtype=np.uint8)
detection_boxes = np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]], dtype=np.float32)
detection_masks = np.ones((2, 15, 15), dtype=np.float32)
detection_classes = np.array([1, 2], dtype=np.int32)
detection_scores = np.array([0.9, 0.8], dtype=np.float32)

# Reproduce the box_ind error
try:
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
        detection_masks, 
        detection_boxes,
        image_np.shape[0], 
        image_np.shape[1]
    )
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, np.uint8)
    
    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        detection_boxes,
        detection_classes,
        detection_scores,
        category_index={1: {'id': 1, 'name': 'person'}, 2: {'id': 2, 'name': 'car'}},
        instance_masks=detection_masks_reframed
    )
except TypeError as e:
    print(f"Error: {e}")
    assert "Got an unexpected keyword argument 'box_ind'" in str(e), "Expected error not found"