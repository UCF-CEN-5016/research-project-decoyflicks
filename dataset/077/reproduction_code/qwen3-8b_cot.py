import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard

def create_model():
    model = tf.keras.Sequential([
        layers.Input(shape=(224, 224, 3)),
        hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5"),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_tensorboard_callback():
    return TensorBoard(log_dir='./logs')

def train_model():
    model = create_model()
    tensorboard = create_tensorboard_callback()

    # Incorrect or non-existent file path
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        'drive/MyDrive/Dog Vision/train/',
        image_size=(224, 224),
        batch_size=32
    )

    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        'drive/MyDrive/Dog Vision/val/',
        image_size=(224, 224),
        batch_size=32
    )

    model.fit(
        train_data,
        epochs=10,
        validation_data=val_data,
        callbacks=[tensorboard]
    )
    return model

# Run the training
model = train_model()

from google.colab import drive
     drive.mount('/content/drive')

train_data = tf.keras.preprocessing.image_dataset_from_directory(
         '/content/drive/MyDrive/Dog Vision/train/',
         image_size=(224, 224),
         batch_size=32
     )