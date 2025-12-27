import tensorflow as tf
from tensorflow import keras
from keras_cv.models import YoloV8
from keras_cv.metrics import COCOMetrics

# Define a minimal YOLOV8 model
model = YoloV8(
    classes=1,
    backbone="resnet50",
    checkpoint=None,
    classifier_activation="sigmoid",
)

# Define a COCO metrics callback with a validation dataset
train_ds = tf.data.Dataset.from_tensor_slices(
    (
        tf.random.uniform((32, 256, 256, 3)),
        {
            "boxes": tf.random.uniform((32, 4, 4)),
            "classes": tf.random.uniform((32, 4), minval=0, maxval=2, dtype="int32"),
        },
    )
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (
        tf.random.uniform((32, 256, 256, 3)),
        {
            "boxes": tf.random.uniform((32, 4, 1, 4)),  # Inconsistent box annotation shape
            "classes": tf.random.uniform((32, 4), minval=0, maxval=2, dtype="int32"),
        },
    )
)

# Compile the model
model.compile(
    loss={
        "box_loss": keras.losses.Huber(),
        "class_loss": keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    },
    optimizer=keras.optimizers.Adam(),
)

# Train the model with the COCO metrics callback
coco_metrics = COCOMetrics(val_ds, "model.h5")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    callbacks=[coco_metrics],
)