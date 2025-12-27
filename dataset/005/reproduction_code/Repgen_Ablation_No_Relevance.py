import tensorflow as tf
from official.vision.delf import delf_model

# Define batch size and image dimensions
batch_size = 2
image_size = 321

# Create random uniform input data
input_data = tf.random.uniform((batch_size, image_size, image_size, 3), minval=0.0, maxval=1.0, seed=0)

# Call the mobilenetV4-conv-s function from tf-models-official with the specified parameters
model = delf_model.Delf(block3_strides=True, name='DELF')
model.init_classifiers(num_classes=1000)
model.trainable = False

# Set to evaluation mode
output = model(input_data, training=False)

# Print output
print(output)