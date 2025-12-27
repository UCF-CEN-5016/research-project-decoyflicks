import tensorflow as tf

# Assuming necessary imports and setup are already done in the environment
from object_detection.models import faster_rcnn_resnet_v1_feature_extractor_tf1_test as feature_extractor_module

def reproduce_bug():
    batch_size = 4
    height, width = 224, 224
    input_data = tf.random.uniform((batch_size, height, width, 3), minval=0, maxval=256, dtype=tf.int32)
    first_stage_features_stride = 16
    architecture = 'resnet_v1_101'
    feature_extractor = feature_extractor_module.FasterRcnnResnetV1FeatureExtractor(
        architecture,
        is_training=False,
        first_stage_features_stride=first_stage_features_stride
    )
    preprocessed_inputs = input_data / 255.0
    rpn_feature_map = feature_extractor.extract_proposal_features(preprocessed_inputs, preprocessed_inputs.get_shape().as_list()[1:3])
    assert rpn_feature_map.shape == (batch_size, height // first_stage_features_stride, width // first_stage_features_stride, 1024)

    # Repeat for different strides and input sizes
    first_stage_features_stride = 8
    rpn_feature_map = feature_extractor.extract_proposal_features(preprocessed_inputs, preprocessed_inputs.get_shape().as_list()[1:3])
    assert rpn_feature_map.shape == (batch_size, height // first_stage_features_stride, width // first_stage_features_stride, 1024)

    input_data_small = tf.random.uniform((1, 112, 112, 3), minval=0, maxval=256, dtype=tf.int32)
    preprocessed_inputs_small = input_data_small / 255.0
    rpn_feature_map_small = feature_extractor.extract_proposal_features(preprocessed_inputs_small, preprocessed_inputs_small.get_shape().as_list()[1:3])
    assert rpn_feature_map_small.shape == (1, 112 // first_stage_features_stride, 112 // first_stage_features_stride, 1024)

reproduce_bug()