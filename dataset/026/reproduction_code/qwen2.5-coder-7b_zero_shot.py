import tensorflow as tf
from object_detection.utils import ops
import numpy as np

# Some example data to reproduce the error
image_np = np.random.rand(256, 512, 3)
detection_masks = np.random.rand(10, 256, 512)

# This line will raise an unexpected keyword argument 'box_ind' error
ops.reframe_box_masks_to_image_masks(detection_masks_reframed=None,
                                     detection_boxes=None,
                                     image_np.shape[1] image_np.shape[2],
                                     box_ind=tf.range(10), 
                                     extrapolation_value=0.0)