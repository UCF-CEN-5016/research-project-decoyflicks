import tensorflow as tf
import keras.layers as KL
import keras.models as KM
import keras.callbacks
import tensorflow_hub as hub
import os

tf.random.set_seed(42)

TRAIN_DIR = "drive/MyDrive/Dog Vision/train/"
VAL_DIR = "drive/MyDrive/Dog Vision/val/"

def create_model():
    input_layer = KL.Input(shape=(224, 224, 3))
    mobilenet_layer = hub.KerasLayer(
        "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5",
        input_shape=(224, 224, 3),
        trainable=True
    )(input_layer)
    output = KL.Dense(120, activation="softmax")(mobilenet_layer)  # example number of classes
    model = KM.Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        labels='inferred',
        label_mode='int',
        batch_size=32,
        image_size=(224, 224),
        shuffle=True
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        VAL_DIR,
        labels='inferred',
        label_mode='int',
        batch_size=32,
        image_size=(224, 224),
        shuffle=False
    )
    model = create_model()
    tensorboard_cb = keras.callbacks.TensorBoard(log_dir="./logs")
    earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    try:
        model.fit(
            train_ds,
            epochs=100,
            validation_data=val_ds,
            validation_freq=1,
            callbacks=[tensorboard_cb, earlystop_cb]
        )
    except tf.errors.NotFoundError as e:
        print("Caught NotFoundError:", e)
        assert "No such file or directory" in str(e)
        assert "drive/MyDrive/Dog Vision/train/17e00d79ad69729522d8705e95939f01.jpg" in str(e)
        raise

if __name__ == "__main__":
    train_model()