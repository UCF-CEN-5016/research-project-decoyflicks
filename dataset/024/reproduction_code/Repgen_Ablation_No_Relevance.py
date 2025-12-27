# Import necessary libraries
from delf.python.extract_features import ExtractFeatures
from absl.flags import FLAGS

# Set up flags and arguments
FLAGS.config_path = 'delf_config_example.pbtxt'
FLAGS.list_images_path = 'list_images.txt'
FLAGS.output_dir = 'data/oxford5k_features'

# Initialize the feature extraction module
feature_extractor = ExtractFeatures()

# Attempt to extract features
try:
    feature_extractor.extract()
except ModuleNotFoundError as e:
    print(e)