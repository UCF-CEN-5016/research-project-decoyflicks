import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# 1. Minimal setup: create dummy train and test datasets
num_train_samples = 100
num_test_samples = 20
num_features = 10
num_classes = 3

# Random data for train and test
x_train = np.random.random((num_train_samples, num_features)).astype(np.float32)
y_train = keras.utils.to_categorical(np.random.randint(num_classes, size=(num_train_samples,)), num_classes)

x_test = np.random.random((num_test_samples, num_features)).astype(np.float32)
y_test = keras.utils.to_categorical(np.random.randint(num_classes, size=(num_test_samples,)), num_classes)

# 2. Create tf.data.Dataset objects for train and test
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(16)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(16)

# 3. Simple model
model = keras.Sequential([
    layers.Input(shape=(num_features,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax'),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Triggering the bug: using test set as validation_data in model.fit
history = model.fit(
    train_dataset,
    epochs=5,
    validation_data=test_dataset  # <-- test set used as validation here
)

# 5. After training, evaluate on the test set again to see possible optimistic bias
test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

# Note:
# Using test_dataset as validation_data during training means the model 'sees' the test data,
# and validation metrics during training will be overly optimistic.
# Proper way: split train data into train/validation or use separate validation set.