def _gather_batched(indices, class_targets, box_targets, index, pos_weights=None):
    # ... 
    classes = tf.gather_nd(class_targets, indices)
    boxes = tf.gather_nd(box_targets, indices)

def _gather_batched(indices, class_targets, box_targets, index, pos_weights=None):
    # ... 
    classes = target_gather._gather_batched(class_targets, indices, pos_weights)
    boxes = target_gather._gather_batched(box_targets, indices, pos_weights)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras_retinanet import models

def build_model(input_shape):
    """Build a simpleRetinaNet model for testing."""
    inputs = layers.Input(shape=input_shape)
    features = mobilenet_v2(inputs, include_top=False)
    
    class predictions = tf.keras.layers.GlobalAveragePooling2D()(features)
    box predictions = tf.keras.layers GlobalAveragePooling2D() (features)
    
    return models.retinanet-retina_net predicting head(classpredictions, boxpredictions)

def _gather_batched(indices, class_targets, box_targets):
    classes = tf.gather_nd(class_targets, indices)
    boxes = tf.gather_nd(box_targets, indices)
    return classes, boxes

# Create a dummy dataset with invalid index
batch_size = 2
num_samples = 3
dummy_images = tf.random.normal((batch_size, 100, 100, 3))
dummy_boxes = [tf.zeros((5,4))] * batch_size  # Each sample has 5 boxes
dummy_classes = [tf.zeros(5) for _ in range(batch_size)]

# Create a model instance (needs to be properly compiled)
model = build_model(dummy_images.shape[1:])

try:
    _, _, _, pos_weights = model(dummy_images.take(2))
except InvalidArgumentError as e:
    print("Invalid argument error occurred:", e)

import tensorflow as tf

def build_model(input_shape):
    """Build a simpleRetinaNet model for testing."""
    inputs = tf.keras.Input(shape=input_shape)
    features = tf.keras.applications.mobilenet_v2(inputs, include_top=False)
    
    class_predictions = tf.keras.layers.GlobalAveragePooling2D()(features)
    box_predictions = tf.keras.layers.GlobalAveragePooling2D()(features)
    
    return tf.keras.Model(inputs=inputs, outputs=[class_predictions, box_predictions])

def _gather_batched(indices, class_targets, box_targets):
    # Validate indices to prevent out-of-bounds access
    valid_indices = tf.where(tf.greater(indices, 0))
    classes = tf.gather_nd(class_targets, valid_indices)
    boxes = tf.gather_nd(box_targets, valid_indices)
    return classes, boxes

# Create a proper dataset with consistent targets per sample
batch_size = 2
num_samples = 3
dummy_images = tf.random.normal((batch_size, 100, 100, 3))
dummy_boxes = [tf.zeros((5,4))] * batch_size  # Each sample has 5 boxes consistently
dummy_classes = [tf.zeros(5) for _ in range(batch_size)]

# Create a model instance (needs to be properly compiled)
model = build_model(dummy_images.shape[1:])

try:
    _, _, _, pos_weights = model(dummy_images.take(2))
except InvalidArgumentError as e:
    print("Invalid argument error occurred:", e)