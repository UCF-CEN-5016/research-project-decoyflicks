import tensorflow as tf
import tensorflow_addons as tfa
from official.vision.beta import train

def main():
    # Set environment variables
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

    # Load dataset
    train_data = tf.data.TFRecordDataset('/ppusw/datasets/vision/imagenet/tfrecords/train*')
    valid_data = tf.data.TFRecordDataset('/ppusw/datasets/vision/imagenet/tfrecords/valid*')

    # Prepare model
    model = train.create_model('resnet_rs_imagenet')

    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train model
    try:
        model.fit(train_data.batch(2), validation_data=valid_data.batch(2), epochs=100)
    except RuntimeError as e:
        print(f"RuntimeError: {e}")

if __name__ == "__main__":
    main()