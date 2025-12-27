import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def create_model():
    model = models.Sequential([
        layers.Input(shape=(256, 256, 3)),
        layers.Conv2D(16, 3, activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(10, activation='softmax')
    ])
    return model

def compile_model(model):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss=loss_fn, optimizer=optimizer)
    return model

def generate_data(batch_size, num_classes, image_size):
    return np.random.rand(batch_size, *image_size), np.random.randint(0, num_classes, (batch_size, 256, 256))

def train_step(model, images, labels, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def evaluate_model(model, data):
    total_loss = 0.0
    for images, labels in data:
        predictions = model(images)
        # You can still compute metrics (like accuracy) here
        # But **no loss is computed** during evaluation
    return total_loss

def main():
    batch_size = 32
    num_classes = 10
    image_size = (256, 256, 3)

    train_data = tf.data.Dataset.from_tensor_slices(generate_data(batch_size, num_classes, image_size))
    val_data = tf.data.Dataset.from_tensor_slices(generate_data(batch_size, num_classes, image_size))

    model = create_model()
    model = compile_model(model)

    for epoch in range(5):
        print(f"\nEpoch {epoch}:")
        
        for images, labels in train_data:
            loss = train_step(model, images, labels, model.loss, model.optimizer)
            print(f"  Training Loss: {loss.numpy():.4f}")

        val_loss = evaluate_model(model, val_data)
        print(f"  Validation Loss: {val_loss:.4f}")

if __name__ == "__main__":
    main()