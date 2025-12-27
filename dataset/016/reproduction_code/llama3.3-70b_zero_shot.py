import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np

# Define the model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model

# Define the dataset
def create_dataset():
    train_images = np.random.rand(1000, 224, 224, 3)
    train_labels = np.random.randint(0, 10, 1000)
    return train_images, train_labels

# Define the multi-worker mirrored strategy
def create_strategy(worker_hosts, task_index):
    strategy = tf.distribute.MultiWorkerMirroredStrategy(
        communication=tf.distribute.experimental.CollectiveCommunication.AUTO,
        worker_hosts=worker_hosts,
        task_index=task_index
    )
    return strategy

# Train the model
def train_model(strategy, model, dataset):
    with strategy.scope():
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    train_images, train_labels = dataset
    model.fit(train_images, train_labels, epochs=10, batch_size=32)

# Main function
def main():
    worker_hosts = 'ip1:6655,ip2:6655'
    task_index = 0
    strategy = create_strategy(worker_hosts, task_index)
    model = create_model()
    dataset = create_dataset()
    train_model(strategy, model, dataset)

if __name__ == '__main__':
    main()