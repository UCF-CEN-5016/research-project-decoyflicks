import numpy as np
import matplotlib.pyplot as plt

def show_batch(batch, num_of_examples):
    # Instead of using num_of_examples to control how many images to show,
    # we are ignoring it and showing all images in the batch.
    # This is where the bug is introduced.
    for i, image in enumerate(batch):
        plt.subplot(1, len(batch), i+1)
        plt.imshow(image)
        plt.axis('off')
    plt.show()

# Simulate a batch of images
batch = [np.random.rand(256, 256, 3) for _ in range(10)]

# Call show_batch with a specific number of examples
show_batch(batch, 5)