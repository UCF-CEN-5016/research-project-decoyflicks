import tensorflow as tf
import matplotlib.pyplot as plt

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(32,)),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Generate some fake data
import numpy as np
x = np.random.random((100, 32))
y = np.random.randint(0, 10, (100,))

# Train the model
model.fit(x, y, epochs=5)

# Plot something
plt.imshow([[1, 2], [3, 4]])
plt.show()