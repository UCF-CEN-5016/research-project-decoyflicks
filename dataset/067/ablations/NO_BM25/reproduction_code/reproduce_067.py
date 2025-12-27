import os
import keras
import keras_cv
import numpy as np
from keras_cv.datasets.pascal_voc.segmentation import load as load_voc

os.environ["KERAS_BACKEND"] = "jax"

train_ds = load_voc(split="sbd_train")
eval_ds = load_voc(split="sbd_eval")

def preprocess_tfds_inputs(inputs):
    def unpackage_tfds_inputs(tfds_inputs):
        return {
            "images": tfds_inputs["image"],
            "segmentation_masks": tfds_inputs["class_segmentation"],
        }

    outputs = inputs.map(unpackage_tfds_inputs)
    outputs = outputs.map(keras_cv.layers.Resizing(height=512, width=512))
    outputs = outputs.batch(4, drop_remainder=True)
    return outputs

train_ds = preprocess_tfds_inputs(train_ds)
eval_ds = preprocess_tfds_inputs(eval_ds)

BATCH_SIZE = 4
NUM_CLASSES = 21

model = keras_cv.models.DeepLabV3Plus.from_preset(
    "resnet50_v2_imagenet", num_classes=NUM_CLASSES
)

model.compile(
    optimizer=keras.optimizers.SGD(
        learning_rate=0.007 * BATCH_SIZE / 16, weight_decay=0.0001, momentum=0.9, clipnorm=10.0
    ),
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=[
        keras.metrics.MeanIoU(num_classes=NUM_CLASSES, sparse_y_true=False, sparse_y_pred=False),
        keras.metrics.CategoricalAccuracy(),
    ],
)

def dict_to_tuple(x):
    import tensorflow as tf
    return x["images"], tf.one_hot(tf.cast(tf.squeeze(x["segmentation_masks"], axis=-1), "int32"), 21)

train_ds = train_ds.map(dict_to_tuple)
eval_ds = eval_ds.map(dict_to_tuple)

model.fit(train_ds, validation_data=eval_ds, epochs=1)

images, masks = next(iter(eval_ds.take(1)))
images = keras.ops.convert_to_tensor(images)
masks = keras.ops.convert_to_tensor(masks)
preds = keras.ops.expand_dims(keras.ops.argmax(model(images), axis=-1), axis=-1)

print("Predicted shape:", preds.shape)
print("Ground truth shape:", masks.shape)