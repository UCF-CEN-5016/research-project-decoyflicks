import tensorflow as tf

# Load pre-trained model and dataset
model = tf.keras.applications.InceptionV3(weights='imagenet')
dataset = tf.data.Dataset.from_tensor_slices((tf.random.normal([100, 224, 224, 3]), tf.range(100)))

# Define a function to show batches
def show_batch(dataset, num_of_examples):  # Unused argument
    for images, labels in dataset.take(num_of_examples):
        print("Batch:", labels)

# Call the function with unused argument
show_batch(dataset, 5)