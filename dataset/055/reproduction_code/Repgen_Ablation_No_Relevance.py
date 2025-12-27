import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow_decision_forests as tfdf

# Set batch size and image dimensions
batch_size = 32
img_height, img_width = 128, 128

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# Preprocess data
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define feature extraction model
feature_extractor = keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2))
])

# Freeze weights
feature_extractor.trainable = False

# Build GBT model
gbt_model = tfdf.keras.GradientBoostedTreesClassifier()

# Train the GBT model
gbt_model.fit(feature_extractor.predict(train_images), train_labels)

# Evaluate the GBT model
loss, accuracy = gbt_model.evaluate(feature_extractor.predict(test_images), test_labels)
print(f"Test Accuracy: {accuracy}")