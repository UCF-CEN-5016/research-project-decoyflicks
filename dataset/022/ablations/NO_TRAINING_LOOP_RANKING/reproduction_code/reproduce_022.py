import tensorflow as tf
from official.vision.ops.augment import RandAugment

batch_size = 32
height, width = 224, 224
input_data = tf.random.uniform((batch_size, height, width, 3))

level_std_0 = 0
rand_augment_0 = RandAugment(magnitude_std=level_std_0)
output_0 = rand_augment_0(input_data)
assert output_0.shape == input_data.shape
std_0 = tf.math.reduce_std(output_0)
assert std_0 == 1

level_std_1 = 1
rand_augment_1 = RandAugment(magnitude_std=level_std_1)
output_1 = rand_augment_1(input_data)
assert output_1.shape == input_data.shape
std_1 = tf.math.reduce_std(output_1)
assert std_1 == 1

print(f"Standard deviation for level_std=0: {std_0.numpy()}")
print(f"Standard deviation for level_std=1: {std_1.numpy()}")