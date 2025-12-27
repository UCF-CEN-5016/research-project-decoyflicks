import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])
    return model

def build_and_compile_model():
    model = create_model()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.SGD(0.001),
                  metrics=['accuracy'])
    return model

def get_dataset():
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    x_train = tf.image.resize(x_train, (224, 224))
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.batch(32).repeat()
    return dataset

def main():
    tf_config = json.loads(os.environ['TF_CONFIG'])
    num_workers = len(tf_config['cluster']['worker'])
    
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        model = build_and_compile_model()
    
    dataset = get_dataset()
    model.fit(dataset, steps_per_epoch=100, epochs=1)

if __name__ == '__main__':
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': ['localhost:12345', 'localhost:23456']
        },
        'task': {'type': 'worker', 'index': 0}
    })
    main()