import tensorflow as tf

# Create a simple model with batch normalization layers
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10)
])

# Compile the model without specifying the loss function
model.compile(optimizer='adam')

# Create dummy data
x = tf.random.normal([128, 32])
y = tf.random.normal([128, 10])

# Attempt to train the model (this will trigger the warning)
model.fit(x, y, epochs=1)