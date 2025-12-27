import tensorflow as tf
import tensorflow_datasets as tfds

# Set up matplotlib for plotting (inline display)
%matplotlib inline

def show_batch images):
    """Show a batch of images with their predicted masks."""
    import matplotlib.pyplot as plt
    import tensorflow_datasets as tfds
    
    plt.figure(figsize=(10, 10))
    len_ = tf.data.experimental.cardinality(images).numpy()
    # Assuming each image has shape (height, width, channels)
    for i in range(3):  # Show first 3 examples
        image = next(iter(images))[0]
        ax = plt.subplot(1, 3, i + 1)
        ax.set_title(f"Image {i+1}")
        ax.imshow(image.numpy())
        plt.grid(False)

# Load a sample dataset (e.g., Pascal VOC for instance segmentation)
dataset, info = tfds.load('pascal_voc/2018', split='train',
                          with_info=True, shuffle_files=False)

# Create a batch
batch_size = 32
batched_dataset = dataset.shuffle(buffer_size=len(dataset))
batched_dataset = batched_dataset.batch(batch_size)

# Call the function without passing all expected arguments
show_batch(batched_dataset)