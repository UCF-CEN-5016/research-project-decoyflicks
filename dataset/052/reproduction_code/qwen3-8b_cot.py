import tensorflow as tf
import numpy as np

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(None,)),  # Input shape: (sequence_length,)
    tf.keras.layers.Embedding(input_dim=1000, output_dim=64),  # (batch_size, sequence_length, 64)
    tf.keras.layers.GlobalAveragePooling1D(),  # (batch_size, 64)
    tf.keras.layers.Dense(10, activation='softmax')  # (batch_size, 10)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Generate dummy data
x = np.random.normal(size=(32, 10))  # (batch_size, sequence_length)
y = np.random.randint(0, 10, size=(32,))  # (batch_size,)

# Train the model
model.fit(x, y, epochs=1)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 3)),  # (batch_size, height, width, channels)
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),  # (batch_size, 32)
    tf.keras.layers.Dense(10, activation='softmax')
])

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(None,)),
    tf.keras.layers.Embedding(input_dim=1000, output_dim=64),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(10, activation='softmax')
])