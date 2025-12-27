import tensorflow as tf
import matplotlib.pyplot as plt

def show_batch(dataset, num_of_examples):
    for images, masks in dataset.take(1):
        plt.figure(figsize=(10, 10))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.axis("off")
        plt.show()

sample_image = tf.random.uniform([224, 224, 3], maxval=255, dtype=tf.int32)
sample_mask = tf.random.uniform([224, 224, 1], maxval=2, dtype=tf.int32)
dataset = tf.data.Dataset.from_tensors((sample_image, sample_mask)).batch(9)

show_batch(dataset, num_of_examples=9)