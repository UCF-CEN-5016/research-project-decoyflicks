import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load dataset
dataset, info = tfds.load('cityscapes', split='train', shuffle_files=True, with_info=True, as_supervised=True)
BATCH_SIZE = 4
SHUFFLE_BUFFER_SIZE = 1024

def preprocess_data(image, label):
    image = tf.image.resize(image, [512, 1024])
    image = tf.image.per_image_standardization(image)
    return image, label

# Apply preprocessing to the dataset
dataset = dataset.map(preprocess_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

# Create a random index for testing
random_idx = tf.random.uniform([], minval=0, maxval=BATCH_SIZE, dtype=tf.int32)

# Iterate through the dataset to get images and masks
images, masks = next(iter(dataset))

# Perform inference on FCN-32S
# Perform inference on FCN-16S
# Perform inference on FCN-8S

# Plot results
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
fig.delaxes(ax[0, 2])
ax[0, 0].set_title("Image")
ax[0, 0].imshow(images[random_idx] / 255.0)
ax[0, 1].set_title("Image with ground truth overlay")
ax[0, 1].imshow(images[random_idx] / 255.0)
ax[0, 1].imshow(masks[random_idx], cmap="inferno", alpha=0.6)
plt.show()