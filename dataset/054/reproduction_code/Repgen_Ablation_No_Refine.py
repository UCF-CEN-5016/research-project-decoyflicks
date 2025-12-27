import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Split the data into training and testing sets
train_size = int(len(x_train) * 0.8)
test_size = len(x_test)
x_val = x_train[train_size:]
y_val = y_train[train_size:]
x_train = x_train[:train_size]
y_train = y_train[:train_size]

# Define the model
model = tf.keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(512, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train,
                    epochs=10,
                    validation_data=(x_val, y_val))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

# Check for NaN values in trainable weights
nan_weights = any(tf.math.is_nan(w).any() for w in model.trainable_weights)
print(f'Contains NaN values in trainable weights: {nan_weights}')