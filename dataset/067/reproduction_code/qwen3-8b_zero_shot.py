import tensorflow as tf
import numpy as np

# 1. Create dummy input and labels
x = np.random.rand(1, 2, 2, 1)  # Input shape (batch, height, width, channels)
y = np.random.randint(0, 3, (1, 2, 2, 1))  # Labels with an extra dimension (batch, height, width, 1)

# 2. Remove the extra dimension from labels
y = y.reshape((1, 2, 2))  # Now shape is (batch, height, width)

# 3. Build model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2, 2, 1)),
    tf.keras.layers.Conv2D(3, (1, 1)),  # Output shape: (batch, height, width, 3)
])

# 4. Compile with sparse_categorical_crossentropy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# 5. Train the model
model.fit(x, y, epochs=1)