import tensorflow as tf

# Sample data with extra dimension in labels
X = tf.random.normal((32, 10, 3))  # (Batch, Height, Width)
y = tf.random.randint(0, 2, (32, 10))  # (Batch, Height)

# Define the model and loss function
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(1, (3, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)
])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Train the model
model.compile(optimizer='adam', loss=loss_fn)
model.fit(X, y, epochs=1)