import tensorflow as tf
from tensorflow.keras import layers, models

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
        layers.BatchNormalization(),
        layers.Dense(10, activation='softmax')
    ])
    return model

def main():
    model = create_model()
    model.compile(optimizer='adam')

    x_train = tf.random.normal([16, 224, 224, 3])
    y_train = tf.random.uniform([16, 10], minval=0, maxval=10, dtype=tf.int32)

    model.fit(x_train, y_train, epochs=1)

if __name__ == "__main__":
    main()