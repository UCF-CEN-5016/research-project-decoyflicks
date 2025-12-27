import unittest
from object_detection.builders import model_builder
import tensorflow as tf  # Import TensorFlow to resolve the undefined variable issue

class TestDetectionModel(unittest.TestCase):
    @unittest.skipIf(model_builder.is_tf2(), "Test case not applicable for TensorFlow 2")
    def test_detection_model(self):
        num_classes = 90
        detection_model = model_builder.build(num_classes=num_classes)
        images = tf.random.normal([1, 300, 300, 3])
        preprocessed_images, shapes = detection_model.preprocess(images)
        prediction_dict = detection_model.predict(preprocessed_images, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        self.assertIsNotNone(detections)

if __name__ == '__main__':
    unittest.main()