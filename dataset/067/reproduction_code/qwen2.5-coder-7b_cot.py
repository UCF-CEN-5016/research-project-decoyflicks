import tensorflow as tf
from tensorflow.keras import layers, Model

def build_segmentation_model(input_shape, num_classes):
    """Build a simple segmentation model with a final softmax Conv2D layer."""
    inputs = layers.Input(shape=input_shape)
    outputs = layers.Conv2D(filters=num_classes, kernel_size=(3, 3), activation='softmax', padding='same')(inputs)
    return Model(inputs=inputs, outputs=outputs)

def ensure_label_shape(labels):
    """Remove an extra trailing channel dimension from label tensors if present."""
    # If labels have shape (batch, H, W, 1), squeeze the last dim to get (batch, H, W)
    return tf.squeeze(labels, axis=-1) if labels.shape.ndims is not None and labels.shape.ndims == 4 else tf.squeeze(labels)

# Configuration
INPUT_SHAPE = (256, 256, 3)
NUM_CLASSES = 10

# Build model
model = build_segmentation_model(INPUT_SHAPE, NUM_CLASSES)

# Example input images (placeholder/dummy data for demonstration)
# Replace with real image batch when using in practice.
images = tf.zeros((1, *INPUT_SHAPE), dtype=tf.float32)

# Get model predictions (probability maps) and a class map for visualization
predicted_prob_maps = model.predict(images)  # shape: (batch, H, W, num_classes)
predicted_prob_maps_tensor = tf.convert_to_tensor(predicted_prob_maps, dtype=tf.float32)
predicted_class_map = tf.argmax(predicted_prob_maps_tensor, axis=-1)  # shape: (batch, H, W)

# Example ground-truth labels (placeholder/dummy data)
# Expected shape: (batch, H, W) with integer class ids
raw_labels = tf.zeros((1, 256, 256, 1), dtype=tf.int64)  # may have an extra channel dimension
y_true = ensure_label_shape(raw_labels)  # shape: (batch, H, W)

# Compute loss using the probability maps (not the argmax) and the prepared ground truth
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
loss = loss_fn(y_true=y_true, y_pred=predicted_prob_maps_tensor)