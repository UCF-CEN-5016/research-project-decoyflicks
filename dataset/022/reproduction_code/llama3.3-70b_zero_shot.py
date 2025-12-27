import numpy as np
import tensorflow as tf

def randaugment(image, num_layers, magnitude):
    for _ in range(num_layers):
        op_to_select = np.random.randint(0, 2)
        if op_to_select == 0:
            magnitude_std = 0.5
            level = np.random.uniform(0.1, 0.9)
            level_std = np.random.uniform(0.1, 0.9)
            magnitude += level * magnitude_std
        else:
            magnitude_std = 0.5
            level = np.random.uniform(0.1, 0.9)
            level_std = np.random.uniform(0.1, 0.9)
            magnitude = level * (magnitude_std + level_std)
    return image

image = tf.random.uniform((256, 256, 3))
num_layers = 2
magnitude = 10

image = randaugment(image, num_layers, magnitude)
print(magnitude)