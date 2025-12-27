import tensorflow as tf

def _generate_detections_v2_class_aware(predicted_boxes, predicted_scores):
    num_detections = tf.shape(predicted_boxes)[0]
    if num_detections > 0:
        boxes = tf.concat([predicted_boxes], axis=0)
    else:
        boxes = tf.concat([], axis=0)
    return boxes

predicted_boxes = tf.constant([], shape=(0, 4))
predicted_scores = tf.constant([], shape=(0,))

boxes = _generate_detections_v2_class_aware(predicted_boxes, predicted_scores)

try:
    sess = tf.Session()
    result = sess.run(boxes)
except ValueError as e:
    print(e)

tf.compat.v1.disable_eager_execution()
with tf.compat.v1.Session() as sess:
    try:
        result = sess.run(boxes)
    except ValueError as e:
        print(e)