import tensorflow as tf
from object_detection.core.freezable_batch_norm_tf2_test import _train_freezable_batch_norm, show_batch

batch_size = 100
input_data = tf.random.normal((batch_size, 10))

# Train the model
model = _train_freezable_batch_norm(input_data, training_mean=5.0, training_var=10.0, use_sync_batch_norm=True)

# Reset batch norm layer
trained_weights = model.get_weights()
model.set_weights(trained_weights)

# Show batch with num_of_examples parameter set to 10
show_batch(model, num_of_examples=10)