import tensorflow as tf
from object_detection import exporter
from object_detection import model_builder
from object_detection.utils import config_util
from object_detection.utils import tf_version
import os

def main():
    model_dir = 'models/centernet_mobilenetv2fp'
    pipeline_config_path = 'models/centernet_mobilenetv2fp/pipeline.config'
    
    class FakeModel:
        def preprocess(self, inputs):
            return tf.identity(inputs), []

        def predict(self, preprocessed_inputs, true_image_shapes):
            return {'image': tf.layers.conv2d(preprocessed_inputs, 3, 1)}

        def postprocess(self, prediction_dict, true_image_shapes):
            return {}

    mock_model = FakeModel()
    inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    preprocessed_inputs, true_image_shapes = mock_model.preprocess(inputs)
    predictions = mock_model.predict(preprocessed_inputs, true_image_shapes)
    mock_model.postprocess(predictions, true_image_shapes)

    try:
        from tensorflow.contrib.quantize.python import graph_matcher
    except ModuleNotFoundError as e:
        print(e)

if __name__ == '__main__':
    main()