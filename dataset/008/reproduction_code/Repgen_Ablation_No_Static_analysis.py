import unittest
import numpy as np
import tensorflow.compat.v1 as tf
from object_detection.models import faster_rcnn_resnet_v1_feature_extractor as faster_rcnn_resnet_v1
from object_detection.utils import tf_version

class FasterRcnnResnetV1FeatureExtractorTest2(unittest.TestCase):
    def _build_feature_extractor(self, architecture_name):
        if tf_version.is_tf2():
            raise ValueError("This test is not compatible with TensorFlow 2.x.")
        return faster_rcnn_resnet_v1.FasterRCNNResNet50FeatureExtractor(
            is_training=True,
            first_stage_features_stride=16,
            batch_norm_trainable=False)

    def test_extract_proposal_features_with_resnet_v1_50(self):
        input_shape = [4, 224, 224, 3]
        preprocessed_inputs = tf.random.uniform(input_shape)
        feature_extractor = self._build_feature_extractor('resnet_v1_50')
        features_shape = feature_extractor.preprocess(preprocessed_inputs).get_shape()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            actual_features_shape = sess.run(features_shape)
            self.assertEqual(actual_features_shape, [4, 14, 14, 2048])

if __name__ == '__main__':
    unittest.main()