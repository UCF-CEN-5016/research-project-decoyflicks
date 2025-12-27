import tensorflow as tf

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2, 2, 1)),
    tf.keras.layers.Conv2D(3, (2, 2), activation='softmax')
])

# Generate one-hot labels (shape: (batch, height, width, num_classes))
labels = tf.keras.utils.to_categorical([[0, 1, 2]], num_classes=3)  # Shape: (1, 2, 2, 3)

# Compile with the correct loss for one-hot labels
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train the model
model.fit(x=tf.random.normal([1, 2, 2, 1]), y=labels)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')