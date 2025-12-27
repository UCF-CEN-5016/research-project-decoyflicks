import numpy as np
from tensorflow.keras.applications import ResNet50

# Set batch size and image dimensions
batch_size = 16
height, width = 224, 224

# Create random uniform input data
input_data = np.random.uniform(low=0.0, high=1.0, size=(batch_size, height, width, 3))

# Preprocess the input data
preprocessed_input = ResNet50.preprocess_input(input_data)

# Initialize SiameseModel (assuming it's already defined)
class SiameseModel:
    def __init__(self):
        pass

    def train_step(self, data):
        # Placeholder method to simulate training step
        return np.random.rand()

# Compile SiameseModel (assuming triplet_loss is a function that computes the loss)
siamese_model = SiameseModel()
siamese_model.train_step(preprocessed_input)

# Generate synthetic triplets (placeholder for actual dataset handling)
anchor, positive, negative = preprocessed_input[:batch_size//3], preprocessed_input[batch_size//3:2*batch_size//3], preprocessed_input[2*batch_size//3:]

# Call train_step on SiameseModel
loss = siamese_model.train_step((anchor, positive, negative))

# Verify NaN values in loss
import numpy as np
print(np.isnan(loss).any())

# Monitor GPU memory usage (placeholder for actual code to check GPU memory)
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())