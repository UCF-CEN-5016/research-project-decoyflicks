import os
import json
import numpy as np
import tensorflow as tf

def setup_multi_worker_config():
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': ['localhost:6655', 'localhost:6656']
        },
        'task': {'type': 'worker', 'index': 0}
    })

def get_dummy_dataset():
    data = np.random.rand(1000, 224, 224, 3)
    labels = np.random.randint(0, 10, (1000,))
    return tf.data.Dataset.from_tensor_slices((data, labels)).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

def train_model(model, dataset):
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    model.fit(dataset, epochs=1)

def main():
    setup_multi_worker_config()

    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    with strategy.scope():
        model = create_model()

    dataset = get_dummy_dataset()
    train_model(model, dataset)

    print("Training completed.")

if __name__ == "__main__":
    main()