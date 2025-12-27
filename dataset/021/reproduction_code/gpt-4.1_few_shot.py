import tensorflow as tf

# Simulate predicted boxes tensor with zero length on axis 0
predicted_boxes = tf.zeros((0, 4), dtype=tf.float32)  # Shape: (0,4)

# Other dummy tensors for concatenation
scores = tf.zeros((0,), dtype=tf.float32)
classes = tf.zeros((0,), dtype=tf.int32)

# Function that tries to concatenate predictions along axis 1
def generate_detections(boxes, scores, classes):
    # List of tensors to concat (simulate what detection_generator might do)
    tensors_to_concat = []
    if tf.shape(boxes)[0] > 0:
        tensors_to_concat.append(boxes)
    if tf.shape(scores)[0] > 0:
        tensors_to_concat.append(tf.expand_dims(scores, axis=1))
    if tf.shape(classes)[0] > 0:
        tensors_to_concat.append(tf.expand_dims(tf.cast(classes, tf.float32), axis=1))
    
    # This will raise ValueError if tensors_to_concat is empty
    detections = tf.concat(tensors_to_concat, axis=1)
    return detections

# This call will raise:
# ValueError: List argument 'values' to 'ConcatV2' Op with length 0 shorter than minimum length 2.
detections = generate_detections(predicted_boxes, scores, classes)
print(detections)