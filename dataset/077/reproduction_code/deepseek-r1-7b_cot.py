import os
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Ensure these paths are correct for your environment
train_dir = "train/dogs_vision"
val_dir = "val/dogs_vision"

def create_model():
    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.experimental.preprocessing.Rescaling(1./255)(inputs)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=outputs)

def create_tensorboard_callback():
    log_dir = "tensorboard_logs"
    os.makedirs(log_dir, exist_ok=True)
    return tf.keras.callbacks TensorBoard(log_dir=log_dir)

model = create_model()
tensorboard_callback = create_tensorboard_callback()

# Ensure the model is compiled before fitting
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit with TensorBoard callback
history = model.fit(
    x=None,
    epochs=10,
    validation_data=None,
    callbacks=[tensorboard_callback, tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)],
)

model.predict([np.random.rand(224, 224, 3)])