import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Step 1: Create a model with output shape (None, 10, 10, 3)
model = models.Sequential([
    layers.Input(shape=(10, 10, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.Conv2D(3, (1, 1), activation='softmax')
])

# Step 2: Generate dummy data with labels having an extra dimension (batch, 10, 10, 1)
images = np.random.rand(1, 10, 10, 1)
labels = np.random.randint(0, 3, (1, 10, 10, 1))

# Step 3: Compile model with SparseCategoricalCrossentropy (expecting labels shape (batch, 10, 10))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Step 4: Attempt to train (this will raise a shape mismatch error)
try:
    model.fit(images, labels, epochs=1)
except Exception as e:
    print("Error occurred due to shape mismatch:", e)

# Fix: Remove the extra dimension from labels
labels = labels.squeeze(axis=-1)  # Now shape is (1, 10, 10)

# Retrain the model
model.fit(images, labels, epochs=1)