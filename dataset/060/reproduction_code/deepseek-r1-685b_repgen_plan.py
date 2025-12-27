import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class DataConfig:
    def __init__(self):
        self.batch_size = 32
        self.img_size = (160, 160)
        self.num_classes = 3
        self.epochs = 1

def generate_synthetic_data(config: DataConfig):
    input_images = tf.random.normal([config.batch_size] + list(config.img_size) + [3], dtype=tf.float32)
    target_masks = tf.random.uniform([config.batch_size] + list(config.img_size), maxval=config.num_classes, dtype=tf.int32)
    return tf.data.Dataset.from_tensor_slices((input_images, target_masks))

def augment_data(image, mask):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, (image.shape[0], image.shape[1])) # Ensure consistent size after potential ops
    
    if tf.random.uniform([]) > 0.5:
        image = tf.image.flip_left_right(image)
    
    # Ensure both image and mask remain float32 for consistency in the pipeline
    # Mask should typically be int32 for sparse_categorical_crossentropy,
    # but if it needs to be float for some operations, cast it back to int32 before loss.
    # For this specific bug, the image type was the problem.
    return image, mask

def build_segmentation_model(config: DataConfig):
    model = keras.Sequential([
        layers.Input(shape=config.img_size + (3,)),
        layers.Conv2D(16, 3, activation='relu', padding='same'),
        layers.Conv2D(config.num_classes, 1, padding='same', activation='softmax') # Added softmax for classification output
    ])
    return model

def main():
    config = DataConfig()

    dataset = generate_synthetic_data(config)

    train_ds = (
        dataset
        .map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = build_segmentation_model(config)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("Attempting to train the model...")
    try:
        model.fit(train_ds, epochs=config.epochs)
        print("Model training completed successfully. Type mismatch bug resolved.")
    except TypeError as e:
        print("TypeError occurred (unexpected after refactoring):")
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")

if __name__ == "__main__":
    main()
