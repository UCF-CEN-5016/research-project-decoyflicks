import tensorflow as tf

def build_conv_softmax_model(input_shape=(28, 28, 1)):
    """Constructs and returns a Sequential model with Conv2D -> GAP -> Softmax."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), input_shape=input_shape),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Softmax()
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

def generate_dummy_dataset(num_samples=100, image_shape=(28, 28, 1), num_classes=10):
    """Generates dummy image data and integer labels."""
    images = tf.random.normal([num_samples, *image_shape])
    labels = tf.random.uniform([num_samples], minval=0, maxval=num_classes, dtype=tf.int32)
    return images, labels

def train_model(model, images, labels, epochs=1):
    """Trains the given model on the provided dataset for a number of epochs."""
    model.fit(images, labels, epochs=epochs)

if __name__ == '__main__':
    model = build_conv_softmax_model()
    images, labels = generate_dummy_dataset()
    train_model(model, images, labels, epochs=1)