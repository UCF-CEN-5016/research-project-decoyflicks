import tensorflow as tf

# Set random seed for reproducibility
tf.random.set_seed(42)

# Define batch size and image dimensions
batch_size = 16
height = 256
width = 256

# Create random uniform input data with shape (batch_size, height, width, 3)
input_data = tf.random.uniform((batch_size, height, width, 3), minval=0, maxval=1, dtype=tf.float32)

# Initialize the MIRNet model
mirnet_model = MIRNet()

# Monitor the memory usage of the GPU during execution
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Call the MIRNet model on the input data and store the output
output = mirnet_model(input_data)

# Log the shape of the intermediate tensors level1_dau_2, level3_dau_2, and skff_