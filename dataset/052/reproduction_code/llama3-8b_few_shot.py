import tensorflow as tf

# Define the model and compile it
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=128),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer="adam", loss="mean_squared_error")

# Create sample data
train_data = tf.random.normal([100, 10000])
train_labels = tf.random.normal([100, 1])

# Train the model (this will raise an error)
model.fit(train_data, train_labels, epochs=10)

print("Model trained successfully!")