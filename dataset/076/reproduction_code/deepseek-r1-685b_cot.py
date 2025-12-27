import os
import tensorflow as tf
from mrcnn import model as modellib
from mrcnn.config import Config

# Minimal configuration
class SimpleConfig(Config):
    NAME = "coco"
    NUM_CLASSES = 1 + 80  # COCO has 80 classes + background
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# Setup paths
MODEL_DIR = os.path.join("logs")
COCO_MODEL_PATH = "mask_rcnn_coco.h5"  # This would be your pretrained weights file

# Create model in inference mode
config = SimpleConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# This will trigger the error
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Save the model in TensorFlow format first if you have control over the weights
   model.save_weights('model_weights.tf', save_format='tf')
   # Then load with:
   model.load_weights('model_weights.tf', by_name=True)