#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.layers import Layer

print(tf.__version__)  # Outputs: 2.13.0

import object_detection


def _resolve_faster_rcnn_model(module):
    """
    Resolve the Faster R-CNN Inception-ResNet-v2 640x640 model attribute
    from the provided object_detection module. Tries several common
    attribute names that packages might expose.
    """
    candidate_names = (
        "model_faster_rcnn_inception_resnet_v2_640x640",
        "faster_rcnn_inception_resnet_v2_640x640",
        "faster_rcnn_inception_resnet_v2_640x640_model",
        "model_faster_rcnn_inception_resnet_v2",
    )
    for name in candidate_names:
        if hasattr(module, name):
            return getattr(module, name)
    return None


# Reference to the Faster R-CNN Inception-ResNet-v2 640x640 model in object_detection
model = _resolve_faster_rcnn_model(object_detection)

# Use compatible layers if any are deprecated in TF2.13.x
# Layer is imported above for compatibility usage when needed.