import tensorflow as tf
import keras_cv
import numpy as np

class DummyDataset(tf.data.Dataset):
    def _generator():
        for _ in range(4):
            yield {
                "images": tf.random.uniform((224, 224, 3)),
                "boxes": np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32),
                "classes": np.array([1, 2], dtype=np.int32),
            }
    def __new__(cls):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature={
                "images": tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                "boxes": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                "classes": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            },
        ).batch(1)

def preprocess(data):
    return {
        "images": data["images"],
        "labels": {
            "boxes": data["boxes"],
            "classes": data["classes"],
        },
    }

train_ds = DummyDataset().map(preprocess)
val_ds = DummyDataset().map(preprocess)

model = keras_cv.models.YOLOv8(
    classes=3,
    bounding_box_format="xyxy",
)

model.compile(
    optimizer="adam",
    box_loss="giou",
    class_loss="focal",
)

# Custom callback to trigger metrics calculation and error
class DummyEvalCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        y_true = [{"boxes": tf.constant([[[0.1, 0.2, 0.3, 0.4],[0.5,0.6,0.7,0.8]]], dtype=tf.float32),
                   "classes": tf.constant([[1,2]], dtype=tf.int32)}]
        y_pred = [{"boxes": tf.constant([[[0.1, 0.2, 0.3, 0.4]]], dtype=tf.float32),
                   "classes": tf.constant([[1]], dtype=tf.int32)}]
        metric = keras_cv.metrics.COCOMeanAveragePrecision(class_ids=[1,2,3])
        metric.update_state(y_true, y_pred)
        metric.result(force=True)

model.fit(train_ds, validation_data=val_ds, epochs=2, callbacks=[DummyEvalCallback()])