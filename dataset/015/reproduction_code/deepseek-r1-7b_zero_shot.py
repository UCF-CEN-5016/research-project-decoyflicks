import tensorflow as tf
from tensorflow.keras import layers

def show_batch(batch_data):
    num_images = 5
    num_of_examples, *rest = batch_data.shape
    plt.figure(figsize=(10,10))
    for i in range(num_images):
        ax = plt.subplot(num_images, 1, i+1)
        ax.imshow(tf.squeeze(rest[i]))
        ax.axis('off')
    plt.show()

# Create a dummy model and dataset to simulate the batch_data
model = tf.keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10)
])
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Create a dummy dataset
dummy_dataset = tf.keras.utilsarray.dummy imagery data for testing (replace this with actual dataset loading)

show_batch(dummy_dataset)