import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow_models import vision
from tensorflow_slim import prefetcher

# Set random seed for reproducibility
np.random.seed(42)

# Define batch size and image dimensions
batch_size = 10
height, width = 32, 32

# Create random normal input data
images = np.random.normal(size=(batch_size, height, width, 3))
labels = np.random.randint(0, 10, size=(batch_size, 1))

# Prefetcher setup
prefetch_queue = prefetcher.prefetch(tf.data.Dataset.from_tensor_slices({'image': images, 'label': labels}).batch(batch_size), capacity=100)

# Dequeue elements from the prefetch queue
dequeued_tensors = [prefetch_queue.dequeue() for _ in range(10)]

# Sample from beta distribution for mixup and cutmix
alpha, beta = 0.2, 0.2

beta_samples = []
for tensors in dequeued_tensors:
    beta_sample = vision.augment.MixupAndCutmix._sample_from_beta(alpha, beta)
    beta_samples.extend(beta_sample.numpy().flatten())

# Plot KDE of generated beta samples
sns.kdeplot(beta_samples, clip=(0, 1))
plt.title('KDE of Generated Beta Samples')
plt.xlabel('Beta Value')
plt.ylabel('Density')
plt.show()