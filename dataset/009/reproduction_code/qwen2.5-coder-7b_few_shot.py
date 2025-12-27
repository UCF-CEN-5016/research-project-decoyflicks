import tensorflow as tf
import numpy as np

class DetectionFromImageModel(tf.keras.Model):
    """Simple detection-like model with a single convolutional layer."""
    def __init__(self):
        super(DetectionFromImageModel, self).__init__()
        self._conv = tf.keras.layers.Conv2D(3, 3, activation='relu')

    def call(self, inputs):
        return self._conv(inputs)

def build_and_compile_detection_model():
    """Instantiate and compile the detection model."""
    model = DetectionFromImageModel()
    model.compile()  # kept intentionally minimal to match original behavior
    return model

def simulate_export(model):
    """
    Simulate an export routine that expects model outputs.
    Attempts to obtain outputs via predict and reports shape or AttributeError.
    """
    try:
        outputs = model.predict(np.zeros((1, 224, 224, 3)))
        print(f"Exported outputs shape: {outputs.shape}")
    except AttributeError as err:
        print(f"Export error: {err}")

if __name__ == "__main__":
    detection_model = build_and_compile_detection_model()
    simulate_export(detection_model)