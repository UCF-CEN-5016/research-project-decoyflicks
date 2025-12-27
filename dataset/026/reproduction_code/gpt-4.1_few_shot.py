import tensorflow as tf

# Create dummy input image batch (1 image, 100x100, 3 channels)
images = tf.random.uniform(shape=[1, 100, 100, 3])

# Define boxes (normalized coordinates)
boxes = tf.constant([[0.1, 0.1, 0.5, 0.5]], dtype=tf.float32)

# Define box indices (for batch images)
box_ind = tf.constant([0], dtype=tf.int32)

# Crop size
crop_size = [50, 50]

# This call will raise:
# TypeError: Got an unexpected keyword argument 'box_ind'
cropped_images = tf.image.crop_and_resize(
    images,
    boxes,
    box_ind=box_ind,  # This keyword arg causes the TypeError on some TF versions
    crop_size=crop_size
)

print(cropped_images.shape)