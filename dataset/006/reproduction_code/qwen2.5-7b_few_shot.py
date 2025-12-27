import tensorflow as tf

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(10)
    ])
    return model

# Create and compile the model
model = create_model()
model.compile(optimizer='adam', loss='mse')

# Create dummy data
x = tf.random.normal([128, 32])
y = tf.random.normal([128, 10])

# Train the model
model.fit(x, y, epochs=1)