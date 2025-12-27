# Import necessary libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy

# Set up minimal environment
tf.config.set_visible_devices([], 'GPU')
tf.config.set_visible_devices([tf.config.list_physical_devices('GPU')[0]], 'GPU')

# Set environment variables
TF_FORCE_GPU_ALLOW_GROWTH = True

# Define constants
BATCH_SIZE = 2
IMG_HEIGHT = 160
IMG_WIDTH = 160

# Load dataset
train_dir = '/ppusw/datasets/vision/imagenet/tfrecords/train*'
validation_dir = '/ppusw/datasets/vision/imagenet/tfrecords/valid*'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Define model
model = ResNet50(weights=None, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), classes=1001)

# Compile model
model.compile(
    optimizer=SGD(momentum=0.9, nesterov=False),
    loss=CategoricalCrossentropy(from_logits=False),
    metrics=[CategoricalAccuracy()]
)

# Define learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1.6,
    decay_steps=100,
    alpha=0.0
)

# Define model training parameters
train_steps = 100
validation_steps = 25000

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=1
)

# Print loss values
print(history.history['loss'])