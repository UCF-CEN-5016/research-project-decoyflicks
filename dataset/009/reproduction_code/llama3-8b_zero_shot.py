import tensorflow as tf
from official.modeling import model_utils

class DetectionFromImageModule(tf.Module):
    def __init__(self):
        pass

model = DetectionFromImageModule()
model_utils.export_inference_code(model)