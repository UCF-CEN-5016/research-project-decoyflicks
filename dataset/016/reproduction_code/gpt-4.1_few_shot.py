import tensorflow as tf
import os

# Simulate multi-worker environment variables (normally set by cluster manager)
os.environ['TF_CONFIG'] = """
{
    "cluster": {
        "worker": ["localhost:12345", "localhost:23456"]
    },
    "task": {"type": "worker", "index": 0}
}
"""

# Create MultiWorkerMirroredStrategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# Simple model and dataset for demonstration
def make_dataset():
    x = tf.random.normal([1000, 32])
    y = tf.random.uniform([1000], maxval=10, dtype=tf.int64)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(1000).batch(32)
    return dataset

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy']
    )

dataset = make_dataset()

# Train model - throughput will not scale across workers
model.fit(dataset, epochs=3)

print("Training complete")