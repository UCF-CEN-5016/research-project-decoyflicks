import tensorflow as tf
from keras_cv import layers as kcv_layers

# Create dummy data
images = tf.random.uniform(shape=(1, 160, 160, 3), dtype=tf.float32)
masks = tf.random.uniform(shape=(1, 160, 160, 1), dtype=tf.int64)

# Create a dataset
dataset = tf.data.Dataset.from_tensor_slices((images, masks))

# Define two augmentation functions that modify the mask type
def augment_mask_float(mask):
    return tf.cast(mask, tf.float32)

def augment_mask_int(mask):
    return mask

# Use RandomChoice to apply either augmentation
random_choice = kcv_layers.RandomChoice(
    [
        lambda x: (x[0], augment_mask_float(x[1])),
        lambda x: (x[0], augment_mask_int(x[1])),
    ],
    num_choices=2
)

# Apply the augmentation to the dataset
processed_dataset = dataset.map(lambda x, y: (x, random_choice([x, y])))

# Now, when iterating, the error should occur
for data in processed_dataset:
    print(data)

def augment_mask_float(mask):
    return tf.cast(mask, tf.float32)

def augment_mask_int(mask):
    return tf.cast(mask, tf.float32)