import tensorflow as tf
import tensorflow_datasets as tfds
from keras_cv import models, callbacks

# Set up minimal environment
tf.keras.backend.set_image_data_format('channels_last')

def preprocess_data(example):
    image = example['image'] / 255.0
    label = example['label']
    return image, label

# Load dataset (example using TensorFlow Datasets)
train_dataset, info = tfds.load('mnist', with_info=True, split='train', batch_size=32)
train_dataset = train_dataset.map(preprocess_data)

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
callbacks_list = [
    callbacks.ModelCheckpoint('model.keras', save_best_only=True),
    callbacks.EarlyStopping(patience=2)
]

history = model.fit(
    train_dataset,
    epochs=1,
    callbacks=callbacks_list
)

for images, labels in train_dataset.take(1):
    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)