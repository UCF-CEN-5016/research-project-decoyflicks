import tensorflow as tf
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
import os
from keras_cv.layers import preprocessing

# Define batch size and image dimensions
BATCH_SIZE = 10
IMG_SIZE = (256, 256)

# Function to get dataset
def get_dataset(batch_size, img_size, input_img_paths, target_img_paths):
    def load_img_masks(input_img_path, target_img_path):
        input_img = tf_io.read_file(input_img_path)
        input_img = tf_io.decode_png(input_img, channels=3)
        input_img = tf_image.resize(input_img, img_size)
        input_img = tf_image.convert_image_dtype(input_img, "float32")

        target_img = tf_io.read_file(target_img_path)
        target_img = tf_io.decode_png(target_img, channels=1)
        target_img = tf_image.resize(target_img, img_size, method="nearest")
        target_img = tf_image.convert_image_dtype(target_img, "uint8")

        # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
        target_img -= 1
        return input_img, target_img

    dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))
    dataset = dataset.map(load_img_masks, num_parallel_calls=tf_data.AUTOTUNE)
    return dataset.batch(batch_size)

# Load Oxford segmentation dataset
input_dir = "images/"
target_dir = "annotations/trimaps/"

input_img_paths = sorted([os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.endswith(".jpg")])
target_img_paths = sorted([os.path.join(target_dir, fname) for fname in os.listdir(target_dir) if fname.endswith(".png") and not fname.startswith(".")])

train_input_img_paths = input_img_paths[:-100]
train_target_img_paths = target_img_paths[:-100]
val_input_img_paths = input_img_paths[-100:]
val_target_img_paths = target_img_paths[-100:]

# Get datasets
train_dataset = get_dataset(BATCH_SIZE, IMG_SIZE, train_input_img_paths, train_target_img_paths)
valid_dataset = get_dataset(BATCH_SIZE, IMG_SIZE, val_input_img_paths, val_target_img_paths)

# Define augmentation function using KerasCV's RandomChoice layer
augment_fn = preprocessing.RandomChoice(
    [
        preprocessing.AugmentationPipeline([preprocessing.RandomFlip()]),
        preprocessing.AugmentationPipeline([preprocessing.RandomRotation(0.1)]),
    ],
    num_samples=2,
    fixed_batch_size=True,
)

# Augment the datasets
augmented_train_ds = (
    train_dataset.shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=tf_data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(lambda x, y: (x, tf.cast(y, tf.float32)))
    .prefetch(buffer_size=tf_data.AUTOTUNE)
)

resized_val_ds = (
    valid_dataset.map(resize_fn, num_parallel_calls=tf_data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(lambda x, y: (x, tf.cast(y, tf.float32)))
    .prefetch(buffer_size=tf_data.AUTOTUNE)
)

# Build model
def get_model(img_size, num_classes):
    # Define your U-Net architecture here
    pass

model = get_model(IMG_SIZE, num_classes=3)
model.summary()

# Train the model
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
model.fit(augmented_train_ds, epochs=15, validation_data=resized_val_ds)