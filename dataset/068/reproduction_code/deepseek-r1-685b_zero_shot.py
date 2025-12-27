import tensorflow as tf
import keras_cv
from keras_cv import bounding_box
from keras_cv.models import YOLOV8Detector
from keras_cv.metrics.object_detection import BoxCOCOMetrics

def create_mock_data():
    images = tf.random.uniform((2, 512, 512, 3))
    boxes = {
        "boxes": tf.constant([
            [[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]],
            [[[0.2, 0.3, 0.4, 0.5]]]
        ], dtype=tf.float32),
        "classes": tf.constant([[[1, 2]], [[1]]], dtype=tf.float32)
    }
    return images, boxes

model = YOLOV8Detector(
    num_classes=2,
    bounding_box_format="xywh"
)
model.compile(
    optimizer="adam",
    metrics=[BoxCOCOMetrics(bounding_box_format="xywh", evaluate_freq=1)]
)

train_ds = tf.data.Dataset.from_tensor_slices(create_mock_data()).batch(1)
val_ds = tf.data.Dataset.from_tensor_slices(create_mock_data()).batch(1)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1
)