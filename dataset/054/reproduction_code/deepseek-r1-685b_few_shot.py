import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Generate synthetic point cloud data
num_samples = 1000
num_points = 1024
num_classes = 10

# Create train/test splits (incorrect usage)
X_train = np.random.rand(num_samples//2, num_points, 3)
y_train = np.random.randint(num_classes, size=num_samples//2)
X_test = np.random.rand(num_samples//2, num_points, 3)
y_test = np.random.randint(num_classes, size=num_samples//2)

# Create datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

# Simple PointNet-like model
inputs = keras.Input(shape=(num_points, 3))
x = layers.Conv1D(64, 1)(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.GlobalMaxPooling1D()(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# Bug: Using test set as validation set
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(train_dataset, 
          epochs=5, 
          validation_data=test_dataset)  # This is the bug - should use separate val set

# Performance will appear artificially high
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.4f}")  # Overly optimistic due to data leakage