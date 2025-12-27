import tensorflow as tf
from orbit.examples.single_task import SingleTaskTrainer

# Define constants
BATCH_SIZE = 640583
LEARNING_RATE = 0.01
NUM_CLASSES = 1000

# Create synthetic dataset function
def create_synthetic_dataset(batch_size):
    height, width = 224, 224
    return tf.data.Dataset.from_tensor_slices((tf.random.normal([batch_size, height, width, 3]), 
                                               tf.random.uniform([batch_size], maxval=NUM_CLASSES, dtype=tf.int32)))

# Define a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# Create an instance of SingleTaskTrainer
train_dataset = create_synthetic_dataset(BATCH_SIZE)
label_key = 'labels'
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
trainer_options = orbit.utils.StandardTrainerOptions(train_steps=100)

trainer = SingleTaskTrainer(
    train_dataset=train_dataset,
    label_key=label_key,
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    metrics=tf.keras.metrics.SparseCategoricalAccuracy()
)

# Monitor the training process
for step in range(100):
    trainer.train_step(next(iter(train_dataset)))