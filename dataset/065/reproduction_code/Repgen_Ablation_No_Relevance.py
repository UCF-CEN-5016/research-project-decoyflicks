import tensorflow as tf

# Importing GradientBoostedTreesModel from TensorFlow Decision Forests
from tensorflow_decision_forests.keras import GradientBoostedTreesModel

batch_size = 32

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

def create_gbt_model(num_trees=50, max_depth=10, min_examples=10, subsample=0.8, validation_ratio=0.2):
    # Creating a Gradient Boosted Trees model
    model = GradientBoostedTreesModel(
        num_trees=num_trees,
        max_depth=max_depth,
        min_examples=min_examples,
        subsample=subsample,
        validation_ratio=validation_ratio,
        task=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # Correcting the task parameter
    )
    model.compile(metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')])
    return model

model = create_gbt_model()

input_data = tf.random.uniform((batch_size, 32, 32, 3), minval=0, maxval=1)
labels = tf.random.uniform((batch_size,), minval=0, maxval=10, dtype=tf.int32)

one_hot_labels = tf.one_hot(labels, depth=10)

predictions = model(input_data)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss = loss_fn(one_hot_labels, predictions)

import numpy as np
assert not np.isnan(loss.numpy()).any()