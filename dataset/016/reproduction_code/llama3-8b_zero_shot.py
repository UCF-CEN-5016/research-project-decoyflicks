import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Set up distribution strategy
strategy = tf.distribute.MirroredStrategy(devices=['GPU:0', 'GPU:1'])

# Create a mirrored model
model = ResNet50(weights='imagenet')
model = strategy.experimentalmirrored(model)

# Load the dataset
train_dir = '/path/to/train/directory'
validation_dir = '/path/to/validation/directory'

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Compile the mirrored model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
strategy.run(model.fit(train_generator, epochs=1, validation_data=validation_generator))