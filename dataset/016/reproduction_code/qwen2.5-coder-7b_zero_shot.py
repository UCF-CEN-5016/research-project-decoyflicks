import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
TRAIN_DIR = '/path/to/train/directory'
VALIDATION_DIR = '/path/to/validation/directory'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 1
GPU_DEVICES = ['GPU:0', 'GPU:1']


def get_distribution_strategy(devices):
    return tf.distribute.MirroredStrategy(devices=devices)


def create_data_generators(train_dir, validation_dir, img_size, batch_size):
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, validation_generator


def build_model_with_strategy(strategy):
    with strategy.scope():
        model = ResNet50(weights='imagenet')
    return model


def compile_model(model):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


def train_model(model, train_generator, validation_generator, epochs):
    model.fit(train_generator, epochs=epochs, validation_data=validation_generator)


def main():
    strategy = get_distribution_strategy(GPU_DEVICES)

    train_gen, val_gen = create_data_generators(
        TRAIN_DIR,
        VALIDATION_DIR,
        IMG_SIZE,
        BATCH_SIZE
    )

    model = build_model_with_strategy(strategy)
    compile_model(model)
    train_model(model, train_gen, val_gen, EPOCHS)


if __name__ == '__main__':
    main()