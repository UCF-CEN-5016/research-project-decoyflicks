import tensorflow as tf
from official.modeling import model_utils

class DetectionFromImage(tf.Module):
    """Minimal TensorFlow Module used for exporting inference code."""
    def __init__(self) -> None:
        super().__init__()

detection_module = DetectionFromImage()
model_utils.export_inference_code(detection_module)