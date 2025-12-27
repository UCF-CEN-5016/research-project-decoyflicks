import os
import tensorflow as tf
from mrcnn import modellib
from mrcnn.config import Config

# Minimal config class
class InferenceConfig(Config):
    NAME = "coco"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80  # COCO dataset

config = InferenceConfig()

MODEL_DIR = os.getcwd()
COCO_MODEL_PATH = "mask_rcnn_coco.h5"  # Assume this file exists

# Create MaskRCNN model in inference mode
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Attempt to load COCO weights (HDF5 format)
# This triggers NotImplementedError in TF 2.x due to legacy weights
model.load_weights(COCO_MODEL_PATH, by_name=True)