import tensorflow as tf
from official.vision.modeling.layers import detection_generator

tf.get_logger().setLevel('ERROR')
batch_size = 1
predicted_boxes = tf.zeros((batch_size, 0, 4))
scores = tf.zeros((batch_size, 0))
classes = tf.zeros((batch_size, 0))
features = {'predicted_boxes': predicted_boxes, 'scores': scores, 'classes': classes}

try:
    detections = detection_generator._generate_detections_v2_class_aware(features)
except ValueError as e:
    print(e)
    assert "List argument 'values' to 'ConcatV2' Op with length 0 shorter than minimum length 2" in str(e)