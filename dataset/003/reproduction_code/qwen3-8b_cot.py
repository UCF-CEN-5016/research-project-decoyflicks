# Step 1: Install required packages
!pip install tensorflow==2.10  # Use a version where 'experimental' is available
!git clone https://github.com/tensorflow/models.git
%cd models/research/object_detection

# Step 2: Install dependencies (ensure Protobuf is compatible)
!pip install --upgrade protobuf==3.20.1  # Example compatible version

# Step 3: Run code that triggers the error
from object_detection.builders import model_builder

# This will raise the AttributeError if TensorFlow version is incompatible