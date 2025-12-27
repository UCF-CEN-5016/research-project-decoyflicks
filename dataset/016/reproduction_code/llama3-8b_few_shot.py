import tensorflow as tf

# Define model and input data
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
input_data = tf.random.normal([32, 224, 224, 3])

# Define distributed training setup
strategy = tf.distribute.MirroredStrategy(devices=['gpu:0', 'gpu:1'])
with strategy.scope():
    model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train model with multiple nodes and GPUs
strategy.run(tf.function(lambda: model.fit(input_data, epochs=1)))

print("Model throughput:", strategy.run(tf.function(lambda: model.evaluate(input_data))))