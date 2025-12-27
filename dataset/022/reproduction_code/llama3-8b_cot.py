import tensorflow as tf
import random

# Minimal setup
img_height = 224
img_width = 224
num_channels = 3
batch_size = 1

# Set up RandAugment operation
augmentor = tf.keras.preprocessing.image RANDAUGMENT(
    height=img_height,
    width=img_width,
    num_channels=num_channels,
    batch_size=batch_size,
    magnitude=random.randint(1, 10),  # trigger bug
    std=random.uniform(0.5, 2.0)  # trigger bug
)

# Wrap final code in a function to reproduce the bug
def reproduce_bug():
    for _ in range(100):  # run multiple iterations to observe the bug
        img = tf.random.normal([batch_size, img_height, img_width, num_channels])
        aug_img = augmentor(img)
        print(aug_img.shape)  # check if std is either 0 or 1

reproduce_bug()