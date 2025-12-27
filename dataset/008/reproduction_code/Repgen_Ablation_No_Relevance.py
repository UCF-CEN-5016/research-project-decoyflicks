import tensorflow as tf
from object_detection import FasterRCNNModelBuilder

batch_size = 4
height = 224
width = 224
channels = 3

input_data = tf.random.uniform((batch_size, height, width, channels), minval=0, maxval=255, dtype=tf.int32)

feature_extractor = FasterRCNNModelBuilder._build_feature_extractor(
    architecture='resnet_v1_101', first_stage_features_stride=16)

proposal_features = feature_extractor.preprocess(input_data)
rpn_feature_map = feature_extractor.extract_proposal_features(proposal_features)

output_shape = rpn_feature_map.get_shape().as_list()
print(output_shape)
assert output_shape == [4, 14, 14, 1024], f"Expected shape: [4, 14, 14, 1024], but got: {output_shape}"