import modellib
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D
from typing import Optional

DEFAULT_MODEL_DIR = 'path_to_model_dir'
DEFAULT_COCO_WEIGHTS = 'path_to_coco_model'
model_config: Optional[object] = None


def build_inference_maskrcnn(weights_path: str, model_dir: str, config: Optional[object] = None) -> modellib.MaskRCNN:
    """
    Construct a Mask R-CNN model in inference mode and load the specified weights.

    Args:
        weights_path: Path to the weights file to load.
        model_dir: Directory where the model logs and checkpoints are stored.
        config: Model configuration object (can be None).

    Returns:
        An instance of modellib.MaskRCNN with weights loaded.
    """
    mask_rcnn = modellib.MaskRCNN(mode="inference", model_dir=model_dir, config=config)
    mask_rcnn.load_weights(weights_path, by_name=True)
    return mask_rcnn


# Instantiate the model at module import time to preserve original behavior
model = build_inference_maskrcnn(DEFAULT_COCO_WEIGHTS, DEFAULT_MODEL_DIR, config=model_config)