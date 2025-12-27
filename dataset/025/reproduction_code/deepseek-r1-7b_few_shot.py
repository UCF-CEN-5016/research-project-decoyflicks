import tensorflow as tf
from tensorflow.keras import layers, preprocessing

# Example of applying augmentation to a dataset
batch_size = 32
height, width = 224, 224

data_dir = tf.keras.utils.get_file('train', 'http://example.com/data')
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    batch_size=batch_size,
    image_size=(height, width),
    shuffle=True
)

# Apply augmentations like rotation and flipping
augmented_ds = train_ds.map(lambda x: preprocessing.augment.random_rotation(x, angle=45))