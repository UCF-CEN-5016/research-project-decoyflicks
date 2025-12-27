import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def show_batch(image_batch, mask_batch, num_of_examples):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(image_batch[i]))
        plt.axis('off')
        plt.imshow(mask_batch[i], cmap='gray', alpha=0.5)
    plt.show()

image_batch = np.random.rand(9, 256, 256, 3)
mask_batch = np.random.rand(9, 256, 256)
show_batch(image_batch, mask_batch, 9)