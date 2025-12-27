import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds

class Config:
    def __init__(self):
        self.batch_size = 32
        self.image_size = (224, 224)
        self.num_classes = 102
        self.epochs = 1
        self.dataset_name = 'oxford_flowers102'
        self.train_split = 'train[:80%]'
        self.test_split = 'train[80%:]'
        self.learning_rate = 0.01 # Default SGD learning rate
        self.momentum = 0.9
        self.label_smoothing = 0.1

def load_and_preprocess_dataset(config: Config):
    def preprocess_for_model(inputs):
        images, labels = inputs["image"], inputs["label"] # Use "image", "label" for tfds.load output
        images = tf.cast(images, tf.float32)
        # Resize images to the target size
        images = tf.image.resize(images, config.image_size)
        # One-hot encode labels
        labels = tf.one_hot(labels, config.num_classes)
        return images, labels

    train_dataset = tfds.load(
        config.dataset_name,
        split=config.train_split,
        shuffle_files=True,
        as_supervised=False # Keep as_supervised=False to get dict with "image" and "label"
    )
    test_dataset = tfds.load(
        config.dataset_name,
        split=config.test_split,
        shuffle_files=False,
        as_supervised=False
    )

    train_dataset = train_dataset.map(preprocess_for_model, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(preprocess_for_model, num_parallel_calls=tf.data.AUTOTUNE)

    train_dataset = train_dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset

def build_and_compile_model(config: Config):
    input_shape = config.image_size + (3,)
    
    # Use tf.keras.applications for EfficientNetV2S or define a custom KerasCV model if needed
    # Assuming the intent was to use a pre-built Keras model and not KerasCV.ImageClassifier.from_preset directly
    # as it might require KerasCV installation and specific setup.
    # For a direct equivalent to `models.ImageClassifier.from_preset("efficientnetv2_s", num_classes=102)`
    # you would typically use KerasCV, but without it being imported, a standard Keras application is safer.
    # If KerasCV is intended, it should be imported: `from keras_cv.models import ImageClassifier`
    
    # Using tf.keras.applications.EfficientNetV2S as a standard alternative
    base_model = tf.keras.applications.EfficientNetV2S(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(config.num_classes, activation='softmax')
    ])

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=config.label_smoothing),
        optimizer=tf.keras.optimizers.SGD(learning_rate=config.learning_rate, momentum=config.momentum),
        metrics=["accuracy"],
    )
    return model

def train_model(model: tf.keras.Model, train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset, config: Config):
    print(f"Training the model for {config.epochs} epoch(s)...")
    model.fit(
        train_dataset,
        epochs=config.epochs,
        validation_data=test_dataset,
    )
    print("Model training complete.")

def main():
    config = Config()

    train_dataset, test_dataset = load_and_preprocess_dataset(config)
    model = build_and_compile_model(config)
    train_model(model, train_dataset, test_dataset, config)

if __name__ == "__main__":
    main()
