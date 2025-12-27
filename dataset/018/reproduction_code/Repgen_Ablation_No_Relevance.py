import tensorflow as tf
from object_detection import is_tf2
from unittest import skipIf

# Clone the TensorFlow models repository from GitHub to /home/user/research/object_detection
# Navigate to the research/object_detection directory
# Create a virtual environment for Python 3.8 using venv
# Activate the virtual environment
# Install pip version 20.1 or higher using python -m ensurepip --upgrade and then python -m pip install pip==20.1
# Set up TensorFlow 1.x by installing tensorflow==1.15.0 using pip
# Navigate to /home/user/research/object_detection/g3doc/tf2.md and follow the instructions for setting up dependencies
# Run 'python -m pip install --use-feature=fast-deps .' as a test if it works instead of 'python -m pip install --use-feature=2020-resolver .'
# If the fast-deps command also fails, attempt to manually set up the required packages using pip

@skipIf(is_tf2(), "This test is for TensorFlow 1.x")
def test_faster_rcnn_resnet_v1_feature_extractor():
    import models.faster_rcnn_resnet_v1_feature_extractor_tf1_test as test_module
    test_module.main()

test_faster_rcnn_resnet_v1_feature_extractor()