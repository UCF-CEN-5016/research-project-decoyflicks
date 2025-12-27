import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Generate synthetic point cloud data
num_samples = 1000
num_points = 1024
num_classes = 10

# Create train/test splits
X = np.random.rand(num_samples, num_points, 3)
y = np.random.randint(num_classes, size=num_samples)

# Create datasets
dataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(num_samples).batch(32)
train_size = int(0.8 * num_samples)
train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

# Simple PointNet-like model
inputs = keras.Input(shape=(num_points, 3))
x = layers.Conv1D(64, 1)(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.GlobalMaxPooling1D()(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# Use a separate validation set
val_size = int(0.1 * train_size)
val_dataset = train_dataset.take(val_size)
train_dataset = train_dataset.skip(val_size)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(train_dataset, 
          epochs=5, 
          validation_data=val_dataset)

# Evaluate model on the test set
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.4f}")