import tensorflow as tf
from object_detection.utils import config  # Or appropriate model imports


class DetectionExporter(tf.Module):
    def __init__(self, model: tf.Module):
        super().__init__()
        self._model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32, name='images')])
    def serve(self, images: tf.Tensor) -> dict:
        outputs = self._model(images)
        return {'detections': outputs}


def export_saved_model(model: tf.Module, export_dir: str = 'exported_model'):
    """
    Wraps a detection model in a tf.Module with a serving signature and saves it as a SavedModel.

    Args:
        model: A trained detection model callable that accepts an images tensor.
        export_dir: Directory where the SavedModel will be written.
    """
    exporter = DetectionExporter(model)
    signature = exporter.serve.get_concrete_function(
        tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32, name='images')
    )
    tf.saved_model.save(exporter, export_dir, signatures=signature)


# Example usage during or after training:
# model = ...  # Your trained Faster R-CNN model
# export_saved_model(model)
#
# Note: Modify exporter_main_v2.py to call export_saved_model(...) where appropriate.