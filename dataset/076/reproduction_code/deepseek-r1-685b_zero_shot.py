import os
import tensorflow as tf
from mrcnn.config import Config
from mrcnn import model as modellib

class TestConfig(Config):
    NAME = "test"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 81

config = TestConfig()
MODEL_DIR = os.path.join(".", "logs")
COCO_MODEL_PATH = "mask_rcnn_coco.h5"

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)