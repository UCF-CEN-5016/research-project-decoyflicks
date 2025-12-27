import tensorflow as tf
from tensorflow_model_optimization.python.core.sparsity.keras import magnitude

data = tf.random.uniform((1, 3, 224, 224))
level_std = 0.5
augment = tf.data.AugmentationRandAugment(level=0.5, level_std=0.5)

for i in range(10):
    augmented_data = augment(data)