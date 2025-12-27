import tensorflow as tf

# Define a simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Define the distribution strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy(
    worker_hosts=['ip1:6655', 'ip2:6655'],
    cluster_spec=tf.distribute.cluster_resolver.SimpleClusterResolver({
        'worker': ['ip1:6655', 'ip2:6655']
    })
)

# Create a dataset
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalize the data
x_train = x_train.reshape(-1, 784) / 255.0

# Define the dataset
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.batch(32)

# Train the model with the distribution strategy
with strategy.scope():
    model.fit(dataset, epochs=10)