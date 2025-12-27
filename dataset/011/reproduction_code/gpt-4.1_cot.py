import tensorflow as tf
import numpy as np

# Enable mixed precision with float16
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Create a minimal ResNet50-like model with swish and batchnorm (simplified)
def create_model(input_shape=(160, 160, 3), num_classes=1001):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    x = tf.keras.applications.resnet.preprocess_input(x)  # optional, can skip
    # Simplified backbone: use tf.keras.applications.ResNet50 with customizations
    base_model = tf.keras.applications.ResNet50(include_top=False, input_tensor=x,
                                                weights=None, pooling='avg')
    x = base_model.output
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)  # output in float32
    model = tf.keras.Model(inputs, outputs)
    return model

# Create model
model = create_model()

# Define optimizer with cosine decay learning rate (initial LR=1.6)
lr_schedule = tf.keras.experimental.CosineDecay(
    initial_learning_rate=1.6,
    decay_steps=100,
    alpha=0.0
)

optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

# Loss function with label smoothing 0.1
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# Prepare synthetic data for batch size 2
batch_size = 2
input_shape = (160, 160, 3)
num_classes = 1001

def generate_synthetic_data(batch_size):
    images = tf.random.uniform((batch_size,) + input_shape, minval=0, maxval=255, dtype=tf.float32)
    labels = tf.random.uniform((batch_size,), minval=0, maxval=num_classes, dtype=tf.int32)
    labels = tf.one_hot(labels, depth=num_classes)
    return images, labels

# Training step function with mixed precision loss scaling
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
        # Scale loss by global batch size if needed (optional)
        scaled_loss = loss
    scaled_loss = optimizer.get_scaled_loss(scaled_loss) if hasattr(optimizer, 'get_scaled_loss') else scaled_loss
    gradients = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = optimizer.get_unscaled_gradients(gradients) if hasattr(optimizer, 'get_unscaled_gradients') else gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Run training loop for 20 steps to observe loss behavior
for step in range(20):
    images, labels = generate_synthetic_data(batch_size)
    loss = train_step(images, labels)
    tf.print(f"Step {step}, Loss: {loss}")