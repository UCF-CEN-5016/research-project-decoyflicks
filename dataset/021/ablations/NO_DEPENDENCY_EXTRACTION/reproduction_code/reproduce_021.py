import tensorflow as tf
from official.projects.triviaqa import prediction

def test_generate_detections():
    predicted_boxes = tf.constant([], shape=(0, 4), dtype=tf.float32)
    predicted_scores = tf.constant([], shape=(0,), dtype=tf.float32)
    predicted_classes = tf.constant([], shape=(0,), dtype=tf.int32)

    try:
        prediction._generate_detections_v2_class_aware(predicted_boxes, predicted_scores, predicted_classes)
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    test_generate_detections()