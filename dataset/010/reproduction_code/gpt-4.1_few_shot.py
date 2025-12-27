import tensorflow as tf
import tensorflow_datasets as tfds

# Minimal dummy dataset that mimics instance segmentation format
def dummy_dataset():
    def _dummy_data(_):
        # Image: 128x128x3, mask: 128x128 (segmentation mask)
        image = tf.random.uniform(shape=(128, 128, 3), dtype=tf.float32)
        mask = tf.random.uniform(shape=(128, 128, 1), maxval=2, dtype=tf.int32)
        return image, mask

    ds = tf.data.Dataset.range(10).map(_dummy_data).batch(2)
    return ds

# Minimal model: simple convnet for segmentation
def get_model():
    inputs = tf.keras.Input(shape=(128, 128, 3))
    x = tf.keras.layers.Conv2D(8, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(1, 1, padding='same')(x)  # logits
    return tf.keras.Model(inputs, x)

# Loss function for segmentation
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model = get_model()

optimizer = tf.keras.optimizers.Adam()

train_ds = dummy_dataset()
val_ds = dummy_dataset()

@tf.function
def train_step(images, masks):
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss = loss_fn(masks, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

@tf.function
def val_step(images, masks):
    logits = model(images, training=False)
    # BUG: Validation loss not computed or returned properly here
    # For example, loss is computed but not returned or accumulated -> leads to zero reported loss
    _ = loss_fn(masks, logits)  # computed but result ignored
    return 0.0  # always zero

# Training loop with evaluation that reports validation loss always zero
for epoch in range(2):
    # Training
    for images, masks in train_ds:
        loss = train_step(images, masks)

    # Validation
    val_losses = []
    for images, masks in val_ds:
        val_loss = val_step(images, masks)
        val_losses.append(val_loss)

    avg_val_loss = tf.reduce_mean(val_losses).numpy()  # always zero due to val_step bug

    print(f"Epoch {epoch+1}, Train loss: {loss.numpy():.4f}, Validation loss: {avg_val_loss:.4f}")