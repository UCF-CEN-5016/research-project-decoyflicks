import tensorflow as tf
from official import DETR, COCODataLoader

# Create synthetic data for input features with specific shapes and types
def create_synthetic_data():
    images = tf.random.normal([32, 800, 800, 3])
    boxes = tf.random.uniform([32, 100, 4], minval=0, maxval=800)
    labels = tf.random.uniform([32, 100, 1], minval=0, maxval=91, dtype=tf.int32)
    return images, boxes, labels

# Create an instance of the DETR class
detr_model = DETR()

# Define a validation dataset using COCODataLoader with dummy data
dummy_data = create_synthetic_data()
validation_dataset = tf.data.Dataset.from_tensor_slices(dummy_data).batch(32)

# Implement a custom function to manually calculate the losses
def build_losses(labels, model_outputs, aux_losses=None):
    del labels, model_outputs
    return tf.constant([np.nan], tf.float32) + (aux_losses if aux_losses is not None else 0.0)

detr_model.build_losses = build_losses

# Run the validation step and capture the logs
logs = detr_model.evaluate(validation_dataset)