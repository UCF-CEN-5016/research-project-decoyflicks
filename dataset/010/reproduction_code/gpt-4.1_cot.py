import tensorflow as tf
import numpy as np

# Minimal dummy dataset for segmentation
def create_dummy_dataset(batch_size=2, image_size=(128, 128), num_classes=3):
    def _generator():
        while True:
            images = np.random.rand(batch_size, *image_size, 3).astype(np.float32)
            # Dummy masks with class indices
            masks = np.random.randint(0, num_classes, size=(batch_size, *image_size, 1)).astype(np.int32)
            yield images, masks
    return tf.data.Dataset.from_generator(
        _generator,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, *image_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, *image_size, 1), dtype=tf.int32)
        )
    ).take(5)  # small dataset

# Dummy segmentation model
class DummySegmentationModel(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(num_classes, 3, padding='same')
    def call(self, x, training=False):
        return self.conv(x)  # logits

# Loss function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Training step
@tf.function
def train_step(model, images, masks, optimizer):
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss = loss_object(masks, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Evaluation step - THIS IS THE CORE: deliberately missing loss computation!
@tf.function
def eval_step(model, images, masks):
    logits = model(images, training=False)
    # Missing loss computation here, so loss not returned or computed
    # This is the core issue causing validation_loss = 0.0
    # Proper would be: loss = loss_object(masks, logits)
    # But we omit it to simulate the bug
    # Return dummy metrics (e.g., accuracy)
    preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(masks[...,0], preds), tf.float32))
    return accuracy  # no loss returned

def main():
    num_classes = 3
    batch_size = 2
    image_size = (128, 128)

    train_ds = create_dummy_dataset(batch_size, image_size, num_classes)
    val_ds = create_dummy_dataset(batch_size, image_size, num_classes)

    model = DummySegmentationModel(num_classes)
    optimizer = tf.keras.optimizers.Adam()

    # Train loop for 3 steps
    for images, masks in train_ds:
        loss = train_step(model, images, masks, optimizer)
        print(f"Train loss: {loss.numpy():.4f}")

    # Eval loop
    total_accuracy = 0.0
    count = 0
    total_val_loss = 0.0  # will remain zero due to no loss calc
    for images, masks in val_ds:
        accuracy = eval_step(model, images, masks)
        total_accuracy += accuracy.numpy()
        # total_val_loss += ??? no loss returned here
        count += 1

    avg_accuracy = total_accuracy / count
    avg_val_loss = total_val_loss / count  # always zero

    print(f"Validation accuracy: {avg_accuracy:.4f}")
    print(f"Validation loss: {avg_val_loss:.4f}  # <-- always zero due to missing loss in eval")

if __name__ == "__main__":
    main()