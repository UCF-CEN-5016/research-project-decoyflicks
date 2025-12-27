import os
os.environ["KERAS_BACKEND"] = "jax"
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import keras_cv

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
tfds.disable_progress_bar()
data, dataset_info = tfds.load('oxford_flowers102', with_info=True)
train_dataset, test_dataset = data['train'], data['test']

def preprocess_for_model(inputs):
    images, labels = inputs["image"], inputs["label"]
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels

train_dataset = train_dataset.map(preprocess_for_model).batch(BATCH_SIZE).prefetch(AUTOTUNE)
test_dataset = test_dataset.map(preprocess_for_model).batch(BATCH_SIZE).prefetch(AUTOTUNE)

def get_model():
    model = keras_cv.models.ImageClassifier.from_preset(
        "efficientnetv2_s", num_classes=dataset_info.features['label'].num_classes
    )
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        optimizer=keras.optimizers.SGD(momentum=0.9),
        metrics=["accuracy"],
    )
    return model

model = get_model()
model.fit(train_dataset, epochs=1, validation_data=test_dataset)