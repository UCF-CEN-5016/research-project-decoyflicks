import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Constants
BATCH_SIZE = 128
NUM_CLASSES = 10
LEARNING_RATE = 0.001
EPOCHS = 10

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Split the dataset into training, validation, and test sets
(train_images, val_images), (train_labels, val_labels) = train_images[:40000], train_images[40000:]
(train_labels, val_labels) = train_labels[:40000], train_labels[40000:]

# Define a function to create a simple Sequential model
def create_model():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(32, 32, 3)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
    return model

# Configure the model for binary classification
model = create_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=EPOCHS, 
                    validation_data=(val_images, val_labels), batch_size=BATCH_SIZE)

# Evaluate the model on the validation data
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f"Test accuracy: {test_acc}")