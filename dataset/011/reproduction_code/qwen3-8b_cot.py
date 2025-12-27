import tensorflow as tf
import numpy as np

# Set up mixed precision configuration
policy = tf.keras.mixed_precision.Policy('float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Define a simple model (similar to ResNet-RS)
class ResNetLikeModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.stem = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3, activation='swish', input_shape=(160, 160, 3)),
            tf.keras.layers.BatchNormalization()
        ])
        self.blocks = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, 3, activation='swish'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, 3, activation='swish'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D()
        ])
        self.head = tf.keras.Sequential([
            tf.keras.layers.Dense(1001, activation='softmax')
        ])

    def call(self, inputs):
        x = self.stem(inputs)
        x = self.blocks(x)
        return self.head(x)

# Create model and optimizer with cosine decay
model = ResNetLikeModel()
optimizer = tf.keras.optimizers.SGD(
    learning_rate=tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1.6, decay_steps=100
    ),
    momentum=0.9
)

# Simulate data loading (using random data for demonstration)
def data_generator():
    while True:
        images = np.random.rand(2, 160, 160, 3).astype(np.float16)
        labels = np.random.randint(0, 1001, size=(2,))
        yield images, labels

# Compile model (without loss scaling)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train loop with high steps_per_loop (simulating user's config)
for step in range(100):
    images, labels = next(data_generator())
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(f"Step {step+1}, Loss: {loss.numpy()}")