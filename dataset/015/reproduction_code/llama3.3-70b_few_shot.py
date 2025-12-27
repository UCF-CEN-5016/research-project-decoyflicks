import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def show_batch(images, labels, num_of_examples):
    # The num_of_examples argument is not used in this function
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].astype('uint8'))
        plt.title(labels[i])
        plt.axis('off')
    plt.show()

# Sample data
images = np.random.randint(0, 256, size=(10, 256, 256, 3), dtype=np.uint8)
labels = ['label1', 'label2', 'label3', 'label4', 'label5', 'label6', 'label7', 'label8', 'label9', 'label10']

# Call the function with unused argument
show_batch(images, labels, 5)