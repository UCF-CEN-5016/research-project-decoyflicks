import os
import json
import numpy as np
import tensorflow as tf

# Simulate multi-worker setup
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['localhost:6655', 'localhost:6656']
    },
    'task': {'type': 'worker', 'index': 0}
})

# Initialize the strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# Create a dummy dataset (not split across workers)
def get_dummy_dataset():
    data = np.random.rand(1000, 224, 224, 3)
    labels = np.random.randint(0, 10, (1000,))
    return tf.data.Dataset.from_tensor_slices((data, labels)).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

# Define the model inside the strategy scope
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

# Train the model (this is a simple fit)
dataset = get_dummy_dataset()
history = model.fit(dataset, epochs=1)

print("Training completed.")