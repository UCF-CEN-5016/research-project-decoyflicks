import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def create_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.Conv2D(3, (1, 1), activation='softmax')
    ])
    return model

def generate_dummy_data(batch_size, image_shape):
    images = np.random.rand(batch_size, *image_shape)
    labels = np.random.randint(0, 3, (batch_size, *image_shape[:2]))
    return images, labels

def train_model(model, images, labels, optimizer='sgd', loss='sparse_categorical_crossentropy', epochs=1):
    model.compile(optimizer=optimizer, loss=loss)
    model.fit(images, labels, epochs=epochs)

# Step 1: Create a model with output shape (None, 10, 10, 3)
model = create_model((10, 10, 1))

# Step 2: Generate dummy data with labels having shape (batch, 10, 10)
images, labels = generate_dummy_data(1, (10, 10, 1))

# Step 3: Compile model with SparseCategoricalCrossentropy
train_model(model, images, labels)

# Step 4: Train the model
train_model(model, images, labels)