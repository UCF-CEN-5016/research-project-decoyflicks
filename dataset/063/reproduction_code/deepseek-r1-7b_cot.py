import tensorflow as tf

   def synthetic_data():
       images = tf.random.uniform((10, 10, 3), dtype=tf.float32)
       labels = tf.constant([i for i in range(10)], dtype=tf.int64)
       return images, labels

   dataset = tf.data.Dataset.from_tensor_slices((images, labels))

def augment_fn(image, label):
       # Ensure dtypes are preserved; no changes made here.
       return image, label

   augmented_dataset = (dataset
                       .shuffle(buffer_size=5)
                       .map(augment_fn)
                       .batch(2))  # Batch the data for testing purposes

unpacked_dataset = tf.data.experimental.unpackage_inputs(augmented_dataset, 
                                                          element_spec=(tf.float32, tf.int64),
                                                          error seeded=False)

augmented_dataset = (dataset
                    .shuffle(buffer_size=5)
                    .map(lambda image, label: (tf.cast(image, tf.float32), tf.cast(label, tf.int64)))
                    .batch(2))

import tensorflow as tf

def synthetic_data():
    images = tf.random.uniform((10, 10, 3), dtype=tf.float32)
    labels = tf.constant([i for i in range(10)], dtype=tf.int64)
    return (images, labels)

dataset = tf.data.Dataset.from_generator(synthetic_data,
                                         output_signature=(tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
                                                          tf.TensorSpec(shape=(None,), dtype=tf.int64)),
                                         stop_at=10)

def augment_fn(image, label):
    return image, label

augmented_dataset = (
    dataset
    .shuffle(buffer_size=5)
    .map(augment_fn)
    .batch(2)
)

# Ensure data types are preserved and compatible
augmented_dataset = augmented_dataset.map(
    lambda image, label: (image, tf.cast(label, tf.float32))
)  # Cast labels to float if needed

unpacked_dataset = tf.data.experimental.unpackage_inputs(augmented_dataset,
                                                       element_spec=(tf.float32, tf.float32),
                                                       error seeded=False)