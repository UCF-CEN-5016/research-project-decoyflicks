import tensorflow as tf
from official.vision.utils import model_garden

def _expand_batch_elements(batch):
    return tf.nest.map_structure(lambda x: tf.expand_dims(x, 0), batch)

def _cast_label_structures(labels):
    class_ids = tf.nest.map_structure(lambda x: tf.cast(x, dtype=tf.int64), labels["class_ids"])
    boxes = tf.nest.map_structure(lambda x: tf.cast(x, dtype=tf.float32), labels["boxes"])
    return class_ids, boxes

def run_template_evaluation_loop(dataset, template_obj):
    for images, labels in dataset:
        template_obj.eval_step(
            _expand_batch_elements(images),
            _cast_label_structures(labels)
        )

def build_and_compile_model():
    model = model_garden.get_model(...)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=model.garden_config.loss,
        metrics=[tf.keras.metrics.MeanName('loss', dtype=tf.float32)]
    )
    return model

def prepare_dataset_from_slices():
    train_ds = tf.data.Dataset.from_tensor_slices(...)
    train_ds = train_ds.map(...).batch(...)

    eval_ds = tf.data.Dataset.from_tensor_slices(...)
    eval_ds = eval_ds.map(...).batch(...)

    return train_ds, eval_ds

# Run evaluation using provided template and dataset
run_template_evaluation_loop(eval_input_dataset, template)

# Reset metrics before continuing
tf.keras.metrics.Mean().reset()

# Build and compile the model
model = build_and_compile_model()

# Prepare training and evaluation datasets
train_input, eval_input = prepare_dataset_from_slices()

# Reset metrics before final evaluation
tf.keras.metrics.Mean().reset()

# Evaluate the model and print validation loss
loss = model.evaluate(eval_input)
print(f"Validation Loss: {loss}")