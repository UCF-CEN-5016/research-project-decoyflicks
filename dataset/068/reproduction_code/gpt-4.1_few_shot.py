import tensorflow as tf
import keras_cv
from tensorflow import keras
import numpy as np

# Minimal dataset with inconsistent ground truth boxes shape
def create_dataset():
    # Each sample has a variable number of boxes with shape (num_boxes, 4)
    # Here intentionally create varying number of boxes per batch item
    images = tf.random.uniform((4, 64, 64, 3))
    # Ground truth boxes for 4 samples, but different shapes in batch dimension cause concat failure
    # For example, 2 boxes in first 2 samples, 1 box in last 2 samples
    boxes_1 = tf.constant([[[0.1, 0.1, 0.2, 0.2],
                            [0.3, 0.3, 0.4, 0.4]],
                           [[0.2, 0.2, 0.3, 0.3],
                            [0.4, 0.4, 0.5, 0.5]],
                           [[0.5, 0.5, 0.6, 0.6]],
                           [[0.6, 0.6, 0.7, 0.7]]], dtype=tf.float32)
    # Pad to make a tensor with shape (4, ?, 4) but ragged along second dimension
    # This is to simulate inconsistent shape problem during concat

    # Classes for each box (dummy, with variable length matching boxes)
    classes_1 = tf.constant([[1, 2], [1, 2], [1], [2]], dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((images, {"boxes": boxes_1, "classes": classes_1}))
    dataset = dataset.batch(4)
    return dataset

# Dummy model that outputs correct shape
inputs = keras.Input(shape=(64, 64, 3))
x = keras.layers.Conv2D(8, 3, activation="relu")(inputs)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(4)(x)  # dummy output
model = keras.Model(inputs, outputs)

# Dummy YOLOv8-like model wrapper to simulate keras_cv.YOLOv8 interface
class DummyYOLOv8(keras.Model):
    def compile(self, **kwargs):
        super().compile(**kwargs)
    def train_step(self, data):
        images, labels = data
        with tf.GradientTape() as tape:
            preds = self(images, training=True)
            loss = tf.reduce_mean(preds)  # dummy loss
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # Simulate updating COCO metrics with inconsistent ground truth boxes shape
        # This triggers the concat error inside keras_cv.metrics.object_detection.BoxCOCOMetrics
        # Use dummy COCO metric to replicate the error
        self.compiled_metrics.update_state(labels, preds)
        return {"loss": loss}

# Use keras_cv COCO metric which internally calls _box_concat and causes concat error
coco_metric = keras_cv.metrics.BoxCOCOMetrics()

yolo = DummyYOLOv8(inputs, outputs)
yolo.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[coco_metric],
)

train_ds = create_dataset()

# Run training - will raise InvalidArgumentError due to concat on inconsistent shapes
yolo.fit(train_ds, epochs=1)