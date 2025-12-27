import tensorflow as tf
from object_detection.models import faster_rcnn_resnet_v1_feature_extractor as feature_extractor

# Initialize variables outside of functions to be used in multiple test cases
architecture = 'resnet_v1_50'
first_stage_features_stride = 16
input_data = tf.random.uniform((4, 224, 224, 3))

@skipIf(tf.executing_eagerly(), "Skip this test for eager execution")
def test_rpn_feature_map_shape():
    # Ensure that the architecture and input data are defined outside the function
    feature_extractor_instance = feature_extractor.FasterRcnnResnetV1FeatureExtractor(
        architecture,
        is_training=False,
        first_stage_features_stride=first_stage_features_stride,
        batch_norm_params=None,
        reuse=tf.compat.v1.AUTO_REUSE)
    
    rpn_feature_map = feature_extractor_instance.extract_proposal_features(input_data, scope='TestScope')
    
    # Assuming self.assertShapeEqual is a method from a testing framework like unittest
    assert_shape_equal([4, 14, 14, 1024], rpn_feature_map)

# Repeat for other architectures
architecture = 'resnet_v1_101'
feature_extractor_instance = feature_extractor.FasterRcnnResnetV1FeatureExtractor(
    architecture,
    is_training=False,
    first_stage_features_stride=first_stage_features_stride,
    batch_norm_params=None,
    reuse=tf.compat.v1.AUTO_REUSE)
rpn_feature_map = feature_extractor_instance.extract_proposal_features(input_data, scope='TestScope')
assert_shape_equal([4, 14, 14, 2048], rpn_feature_map)

architecture = 'resnet_v1_152'
feature_extractor_instance = feature_extractor.FasterRcnnResnetV1FeatureExtractor(
    architecture,
    is_training=False,
    first_stage_features_stride=first_stage_features_stride,
    batch_norm_params=None,
    reuse=tf.compat.v1.AUTO_REUSE)
rpn_feature_map = feature_extractor_instance.extract_proposal_features(input_data, scope='TestScope')
assert_shape_equal([4, 14, 14, 2048], rpn_feature_map)

def assert_shape_equal(expected, actual):
    # Implement the logic for assertShapeEqual here
    # This is a placeholder function and should be replaced with the actual implementation
    pass