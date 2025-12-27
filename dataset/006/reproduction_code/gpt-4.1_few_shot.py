import tensorflow as tf

# Minimal custom model with non-trainable layer to simulate disconnected gradients
class CustomModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(10)
        self.non_trainable = tf.keras.layers.Lambda(lambda x: tf.stop_gradient(x))  # Breaks gradient flow
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.non_trainable(x)  # No gradients flow beyond here
        return self.dense2(x)

model = CustomModel()

# Compile without specifying loss intentionally (or with disconnected output)
model.compile(optimizer='adam')

# Dummy data
X = tf.random.normal((8, 4))
y = tf.random.normal((8, 1))

# This will produce warnings about missing gradients
model.fit(X, y, epochs=2)