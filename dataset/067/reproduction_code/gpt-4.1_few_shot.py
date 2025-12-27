import tensorflow as tf
from tensorflow import keras
import numpy as np

# Dummy model for classification with 3 classes
model = keras.Sequential([
    keras.layers.Input(shape=(4,)),
    keras.layers.Dense(3)
])

# SparseCategoricalCrossentropy expects labels shape (batch,) not (batch, 1)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Data: 5 samples, labels have an extra last dimension (incorrect)
x = np.random.random((5, 4)).astype(np.float32)
y_wrong = np.array([[0], [1], [2], [1], [0]])  # shape (5,1), should be (5,)

# Data: correct labels shape (squeezed)
y_correct = np.squeeze(y_wrong, axis=-1)

optimizer = keras.optimizers.SGD(learning_rate=0.1)

# Training step showing incorrect usage
with tf.GradientTape() as tape:
    logits = model(x)
    # Passing labels with extra dimension causes poor training or error
    loss_wrong = loss_fn(y_wrong, logits)
grads = tape.gradient(loss_wrong, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))

print(f"Loss with labels shape {y_wrong.shape}: {loss_wrong.numpy()}")

# Training step with correct labels shape
with tf.GradientTape() as tape:
    logits = model(x)
    loss_correct = loss_fn(y_correct, logits)
grads = tape.gradient(loss_correct, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))

print(f"Loss with labels shape {y_correct.shape}: {loss_correct.numpy()}")