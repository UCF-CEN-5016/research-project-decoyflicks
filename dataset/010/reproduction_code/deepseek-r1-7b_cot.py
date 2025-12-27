for images, labels in eval_input_dataset:
    template.eval_step(
        tf.nest.map_structure(lambda x: tf.expand_dims(x, 0), images),
        (tf.nest.map_structure(lambda x: tf.cast(x, dtype=tf.int64), labels["class_ids"]), 
         tf.nest.map_structure(lambda x: tf.cast(x, dtype=tf.float32), labels["boxes"])))

tf.keras.metrics.Mean().reset()

import tensorflow as tf
from official.vision.utils import model_garden

# Define the model
model = model_garden.get_model(...)

# Compile the model with loss function and optimizer
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=model.garden_config.loss,
    metrics=[tf.keras.metrics.MeanName('loss', dtype=tf.float32)]
)

# Prepare training data (ensure it's correctly set up)
train_input = tf.data.Dataset.from_tensor_slices(...)
train_input = train_input.map(...).batch(...)

eval_input = tf.data.Dataset.from_tensor_slices(...)
eval_input = eval_input.map(...).batch(...)

# Reset metrics before evaluation
tf.keras.metrics.Mean().reset()

# Evaluate the model
loss = model.evaluate(eval_input)
print(f"Validation Loss: {loss}")