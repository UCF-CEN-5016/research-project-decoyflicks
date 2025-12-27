# If you are facing an issue with TF2 models and encounter Value errors,
# ensure your TensorFlow version matches the model requirements.

try:
    import tensorflow as tf  # Ensure you have the correct version installed
except ImportError:
    print("TensorFlow must be imported first.")
    exit()

import os

# Install required packages that might be missing for compatibility
os.system('pip install --ignore-pip-system-reinstall protobuf>=3.10.2 numpy')

try:
    from official import model_garden.object_detection exporting model  # Use the exact model path
except ImportError as e:
    print(f"Error: {e}. Check if you have the correct TF2 version installed.")