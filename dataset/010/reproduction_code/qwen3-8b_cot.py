import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# 🧠 Define a simple model for segmentation
model = models.Sequential([
    layers.Input(shape=(256, 256, 3)),
    layers.Conv2D(16, 3, activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(10, activation='softmax')
])

# 🧮 Define the loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

# 🧪 Dummy data for training and validation
batch_size = 32
num_classes = 10
image_size = (256, 256, 3)

train_data = tf.data.Dataset.from_tensor_slices(
    (np.random.rand(batch_size, *image_size), 
     np.random.randint(0, num_classes, (batch_size, 256, 256)))
)

val_data = tf.data.Dataset.from_tensor_slices(
    (np.random.rand(batch_size, *image_size), 
     np.random.randint(0, num_classes, (batch_size, 256, 256)))
)

# 🚀 Training step that computes and applies gradients
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# 📊 Evaluation step that **does not** compute the loss
def evaluate():
    total_loss = 0.0
    for images, labels in val_data:
        predictions = model(images)
        # You can still compute metrics (like accuracy) here
        # But **no loss is computed** during evaluation
    return total_loss

# 📅 Training and evaluation loop
for epoch in range(5):
    print(f"\nEpoch {epoch}:")

    # 🔁 Training loop
    for images, labels in train_data:
        loss = train_step(images, labels)
        print(f"  Training Loss: {loss.numpy():.4f}")

    # 🧪 Evaluation loop (no loss computed)
    val_loss = evaluate()
    print(f"  Validation Loss: {val_loss:.4f}")