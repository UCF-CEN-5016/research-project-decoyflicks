import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
import os

# Set up minimal environment
train_dir = 'drive/MyDrive/Dog Vision/train'
validation_dir = 'drive/MyDrive/Dog Vision/validation'

# Create a model
def create_model():
    url = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5"
    model = tf.keras.Sequential([
        hub.KerasLayer(url, input_shape=(224, 224, 3)),
        layers.Dense(128, activation='relu'),
        layers.Dense(2)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

# Create TensorBoard callback
def create_tensorboard_callback():
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    return tensorboard_callback

# Define training function
def train_model():
    model = create_model()
    tensorboard = create_tensorboard_callback()

    # Fit the model (using non-existent file path)
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=32,
        image_size=(224, 224)
    )
    validation_data = tf.keras.preprocessing.image_dataset_from_directory(
        validation_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=32,
        image_size=(224, 224)
    )
    model.fit(train_data, epochs=10, validation_data=validation_data, callbacks=[tensorboard])

# Run the training function
train_model()