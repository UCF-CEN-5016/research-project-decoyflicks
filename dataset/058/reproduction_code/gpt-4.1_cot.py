import tensorflow as tf
import keras_cv

# Setup: single dummy image (1, 256, 256, 3)
dummy_image = tf.zeros((1, 256, 256, 3), dtype=tf.float32)

# Dummy labels with bounding boxes and class ids
# Format expected by keras_cv RetinaNet:
# {"boxes": [N, 4], "classes": [N]} with classes as integers from 0 to num_classes-1
dummy_labels = {
    "boxes": tf.constant([[[0.1, 0.1, 0.5, 0.5]]], dtype=tf.float32),  # normalized xyxy format
    "classes": tf.constant([[0]], dtype=tf.int32),  # class id 0
}

# Create tf.data.Dataset yielding (image, labels)
def gen():
    while True:
        yield dummy_image[0], {k: v[0] for k, v in dummy_labels.items()}

dataset = tf.data.Dataset.from_generator(
    gen,
    output_signature=(
        tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
        {
            "boxes": tf.TensorSpec(shape=(1, 4), dtype=tf.float32),
            "classes": tf.TensorSpec(shape=(1,), dtype=tf.int32),
        },
    ),
).batch(1)

# Create RetinaNet model with zero classes or mismatched classes
# Here, num_classes=0 to trigger the error
model = keras_cv.models.RetinaNet(
    classes=0,
    bounding_box_format="xyxy",
    backbone="resnet50",
)

# Compile model
model.compile(
    optimizer="adam",
    classification_loss="focal",
    box_loss="smoothl1",
)

# Trigger the error by fitting
model.fit(dataset.take(1), epochs=1)