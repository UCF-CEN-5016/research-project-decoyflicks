import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Minimal code example using KerasCV - note that standard keras imports might be needed if installed via pip
base_model = EfficientNetB0(weights='imagenet', include_top=False)
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

datagen = ImageDataGenerator()
train_generator = datagen.flow_from_directory('path/to/training',
flow_name='training',
class_mode='binary',
target_size=(224, 224))

model.fit(train_generator, epochs=10)