import tensorflow as tf
import tensorflow_datasets as tfds
from keras_cv import models, callbacks

# Set up minimal environment
tf.keras.backend.set_image_data_format('channels_last')

# Load dataset (example using TensorFlow Datasets)
dataset_value = 'mnist'  # Replace with the actual dataset name
dataset, info = tfds.load(dataset_value, with_info=True, split='train', batch_size=32)
dataset = dataset.map(lambda x: (x['image'] / 255.0, x['label']))

# Define a simple model (example)
model = models.ImageClassifier(
    num_classes=10,
    input_shape=(28, 28, 1),
    backbone='efficientnet_b0',
    include_top=True,
    weights=None
)

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Trigger the bug by fitting the model
history = model.fit(
    dataset,
    epochs=1,
    callbacks=[
        callbacks.ModelCheckpoint('model.keras', save_best_only=True),
        callbacks.EarlyStopping(patience=2)
    ]
)

for images, labels in dataset.take(1):
       print("Image shape:", images.numpy().shape)
       print("Label shape:", labels.numpy().shape)