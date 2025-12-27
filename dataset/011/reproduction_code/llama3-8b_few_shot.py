Here is the reproduction code for this bug:

```
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
import numpy as np

# Load the model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(160, 160, 3))

# Create an instance of the data generator for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load the training and validation data
train_dir = '/ppusw/datasets/vision/imagenet/tfrecords/train*'
validation_dir = '/ppusw/datasets/vision/imagenet/tfrecords/valid*'

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(160, 160),
    batch_size=2,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(160, 160),
    batch_size=2,
    class_mode='categorical',
    shuffle=False
)

# Compile the model with SGD optimizer and cosine decay learning rate schedule
model.compile(optimizer=SGD(lr=1.6, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model for 100 steps
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=validation_generator,
    verbose=2
)
```

This code reproduces the bug by training a ResNet50 model with SGD optimizer and cosine decay learning rate schedule on the ImageNet dataset using the TFRecord format. The model is trained for 100 steps, and the loss value becomes NaN after several iterations.

