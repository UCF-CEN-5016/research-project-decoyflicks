import matplotlib.pyplot as plt
import numpy as np

def show_batch(batch, num_of_examples):
    # num_of_examples argument is unused
    for image in batch:
        plt.imshow(np.array(image))
        plt.show()

# Sample batch of images (dummy data)
dummy_batch = [np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8) for _ in range(3)]

# Call show_batch with num_of_examples argument
show_batch(dummy_batch, num_of_examples=2)