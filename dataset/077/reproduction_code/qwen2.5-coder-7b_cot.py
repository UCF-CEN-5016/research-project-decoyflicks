import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
TRAIN_DIR = "train/dogs_vision"
VAL_DIR = "val/dogs_vision"
LOG_DIR = "tensorboard_logs"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

def build_model(input_shape=(224, 224, 3)):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255.0)(inputs)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def make_tensorboard_callback(log_dir=LOG_DIR):
    os.makedirs(log_dir, exist_ok=True)
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir)

def create_data_generators(train_dir, val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE):
    train_datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary"
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary"
    )
    return train_generator, val_generator

def main():
    model = build_model()
    tensorboard_cb = make_tensorboard_callback()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    train_gen, val_gen = create_data_generators(TRAIN_DIR, VAL_DIR)

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=[
            tensorboard_cb,
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
        ],
    )

    sample_input = np.random.rand(1, IMG_SIZE[0], IMG_SIZE[1], 3)
    predictions = model.predict(sample_input)
    print(predictions)

if __name__ == "__main__":
    main()