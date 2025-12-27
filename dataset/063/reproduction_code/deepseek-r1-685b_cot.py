import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

# Setup minimal environment
BATCH_SIZE = 8
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_SIZE = (160, 160)

# Create synthetic dataset matching Oxford Pets structure
def create_synth_data():
    images = tf.random.uniform((100, *IMG_SIZE, 3), dtype=tf.float32)
    masks = tf.random.uniform((100, *IMG_SIZE, 1), maxval=3, dtype=tf.int64)
    return tf.data.Dataset.from_tensor_slices((images, masks))

# Augmentation function with RandAugment
def augment_fn(image, mask):
    # Apply same geometric transforms to both
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    
    # Only apply color jitter to image
    aug = preprocessing.RandomContrast(factor=0.2)
    image = aug(image)
    
    return image, mask

def unpackage_inputs(image, mask):
    return {"image": image, "mask": mask}

# Create and process dataset
train_ds = create_synth_data()

# Triggering condition - same pipeline as bug report
augmented_train_ds = (
    train_ds.shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)
    .prefetch(buffer_size=AUTOTUNE)
)

# Force execution to trigger error
for batch in augmented_train_ds.take(1):
    print(batch["image"].shape, batch["mask"].shape)