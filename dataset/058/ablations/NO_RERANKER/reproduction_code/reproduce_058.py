import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

# Assuming these functions are defined elsewhere in the codebase
def swap_xy(bbox):
    # Placeholder for the actual implementation
    return bbox

def random_flip_horizontal(image, bbox):
    # Placeholder for the actual implementation
    return image, bbox

def resize_and_pad_image(image):
    # Placeholder for the actual implementation
    return image, tf.shape(image), None

def convert_to_xywh(bbox):
    # Placeholder for the actual implementation
    return bbox

class LabelEncoder:
    # Placeholder for the actual implementation
    def encode_batch(self, batch):
        return batch

def get_backbone():
    # Placeholder for the actual implementation
    return None

class RetinaNetLoss(tf.keras.losses.Loss):
    # Placeholder for the actual implementation
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

class RetinaNet(tf.keras.Model):
    # Placeholder for the actual implementation
    def __init__(self, num_classes, backbone):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone

batch_size = 2
num_classes = 80
learning_rate = 0.001
learning_rate_boundaries = [125, 250, 500, 240000, 360000]

def preprocess_data(sample):
    image = sample["image"]
    bbox = swap_xy(sample["objects"]["bbox"])
    class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)
    image, bbox = random_flip_horizontal(image, bbox)
    image, image_shape, _ = resize_and_pad_image(image)
    bbox = tf.stack([
        bbox[:, 0] * image_shape[1],
        bbox[:, 1] * image_shape[0],
        bbox[:, 2] * image_shape[1],
        bbox[:, 3] * image_shape[0],
    ], axis=-1)
    bbox = convert_to_xywh(bbox)
    return image, bbox, class_id

(train_dataset, val_dataset), dataset_info = tfds.load(
    "coco/2017", split=["train", "validation"], with_info=True, data_dir="data"
)

autotune = tf.data.AUTOTUNE
train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)
train_dataset = train_dataset.shuffle(16)
train_dataset = train_dataset.padded_batch(
    batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
label_encoder = LabelEncoder()
train_dataset = train_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
train_dataset = train_dataset.prefetch(autotune)

val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
val_dataset = val_dataset.padded_batch(
    batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
val_dataset = val_dataset.prefetch(autotune)

resnet50_backbone = get_backbone()
loss_fn = RetinaNetLoss(num_classes)
model = RetinaNet(num_classes, resnet50_backbone)

optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate, momentum=0.9)
model.compile(loss=loss_fn, optimizer=optimizer)

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join("retinanet/", "weights" + "_epoch_{epoch}"),
        monitor="loss",
        save_best_only=False,
        save_weights_only=True,
        verbose=1,
    )
]

# The following line is where the bug is expected to be reproduced
model.fit(
    train_dataset.take(20),
    validation_data=val_dataset.take(20),
    epochs=1,
    callbacks=callbacks_list,
    verbose=1,
)