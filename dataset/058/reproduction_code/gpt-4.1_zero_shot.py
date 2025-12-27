import tensorflow as tf
import keras_cv

def gen():
    # Single sample with empty boxes and labels, causing label encoder issue
    yield {
        "image": tf.zeros([128, 128, 3], dtype=tf.float32),
        "labels": tf.constant([], shape=(0,), dtype=tf.int32),
        "boxes": tf.constant([], shape=(0, 4), dtype=tf.float32),
    }

train_ds = tf.data.Dataset.from_generator(
    gen,
    output_signature={
        "image": tf.TensorSpec(shape=(128, 128, 3), dtype=tf.float32),
        "labels": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        "boxes": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
    },
).batch(1)

model = keras_cv.models.RetinaNet(classes=3)

model.compile(
    classification_loss="focal",
    box_loss="smoothl1",
    optimizer="adam",
)

model.fit(train_ds, epochs=1)