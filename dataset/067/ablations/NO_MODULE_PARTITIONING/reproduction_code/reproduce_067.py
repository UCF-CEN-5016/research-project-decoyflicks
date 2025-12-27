import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set TensorFlow version
tf.__version__ = '2.10.0'

# Define base path for the dataset
BASE_PATH = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

# Download and extract the dataset
dataset_path = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', BASE_PATH, extract=True)

# Load the dataset
train_dataset = keras.preprocessing.image_dataset_from_directory(
    'cats_and_dogs_filtered/train',
    image_size=(128, 128),
    batch_size=32
)
val_dataset = keras.preprocessing.image_dataset_from_directory(
    'cats_and_dogs_filtered/validation',
    image_size=(128, 128),
    batch_size=32
)

# Define number of classes
num_classes = 3

# Create the model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(num_classes)
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Prepare labels with an extra dimension
labels = tf.keras.utils.to_categorical(train_dataset.labels, num_classes)
labels = tf.expand_dims(labels, axis=-1)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# Verify model output shape
print(model.output_shape)

# Check loss value
loss = model.evaluate(val_dataset)
print("Loss:", loss)

# Log model summary
model.summary()

# Print predictions
predictions = model.predict(val_dataset)
print(predictions.shape)