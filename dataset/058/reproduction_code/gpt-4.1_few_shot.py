import tensorflow as tf
import keras_cv
from keras_cv.models import RetinaNet

# Create a minimal dataset mimicking the expected structure but with empty labels
def generate_empty_labels_dataset():
    # One sample with an image and empty bounding boxes and classes
    image = tf.random.uniform(shape=(640, 640, 3), dtype=tf.float32)
    # Empty bounding boxes: shape (0, 4)
    boxes = tf.zeros((0, 4), dtype=tf.float32)
    # Empty classes: shape (0,)
    classes = tf.zeros((0,), dtype=tf.int32)
    target = {
        "boxes": boxes,
        "classes": classes,
    }
    return tf.data.Dataset.from_tensors((image, target))

train_ds = generate_empty_labels_dataset().batch(1)

# Instantiate RetinaNet with COCO backbone and 1 class (for simplicity)
model = RetinaNet.from_preset("resnet50_coco", num_classes=1)

# Try to fit the model on dataset with empty labels: should raise InvalidArgumentError
model.compile(
    optimizer="adam",
    classification_loss="focal",
    box_loss="smoothl1",
)

# This will reproduce the error due to empty labels
model.fit(train_ds, epochs=1)