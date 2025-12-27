import tensorflow as tf
from official.vision.beta.modeling import resnet_rs_model
from official.vision.beta.tasks import image_classification
import numpy as np

# Minimal settings similar to reported config
BATCH_SIZE = 2
INPUT_SIZE = 160
NUM_CLASSES = 1001
DTYPE = tf.float16  # Mixed precision dtype as in bug report

# Enable mixed precision globally
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Create a simple ResNet-RS-50 backbone model
def get_model():
    model = resnet_rs_model.ResNetRS(
        depth_multiplier=1.0,
        model_id=50,
        num_classes=NUM_CLASSES,
        dropout_rate=0.25,
        input_specs=tf.keras.layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3), dtype=DTYPE),
        norm_activation=None,
        stochastic_depth_drop_rate=0.0,
        se_ratio=0.25,
        replace_stem_max_pool=True,
        resnetd_shortcut=True,
        stem_type='v1',
        name='resnet_rs50',
    )
    return model

# Dummy data generator with random inputs and labels
def get_dataset():
    # Random float16 images
    images = tf.random.normal([BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, 3], dtype=DTYPE)
    # Random integer labels in range [0, NUM_CLASSES)
    labels = tf.random.uniform([BATCH_SIZE], maxval=NUM_CLASSES, dtype=tf.int32)
    return tf.data.Dataset.from_tensors((images, labels)).repeat()

# Loss function with label smoothing and one-hot encoding
def get_loss():
    return tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1, from_logits=True)

# Training step function
@tf.function
def train_step(model, optimizer, loss_fn, images, labels):
    labels_one_hot = tf.one_hot(labels, depth=NUM_CLASSES)
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss = loss_fn(labels_one_hot, logits)
        # Add weight decay loss
        loss += sum(model.losses)
    scaled_loss = loss
    scaled_loss = optimizer.get_scaled_loss(scaled_loss) if hasattr(optimizer, 'get_scaled_loss') else loss
    gradients = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = optimizer.get_unscaled_gradients(gradients) if hasattr(optimizer, 'get_unscaled_gradients') else gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def main():
    # Build model and optimizer
    model = get_model()
    loss_fn = get_loss()
    # SGD optimizer with initial LR=1.6 as in bug report
    optimizer = tf.keras.optimizers.SGD(learning_rate=1.6, momentum=0.9)
    # Wrap optimizer with LossScaleOptimizer for float16 mixed precision
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    dataset = get_dataset()
    iterator = iter(dataset)

    for step in range(20):
        images, labels = next(iterator)
        loss = train_step(model, optimizer, loss_fn, images, labels)
        tf.print(f"Step {step}, Loss:", loss)
        if tf.math.is_nan(loss):
            tf.print("NaN loss detected at step", step)
            break

if __name__ == "__main__":
    main()