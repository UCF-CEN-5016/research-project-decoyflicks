import os
import sys
import numpy as np
from mrcnn.config import Config
from mrcnn import model as modellib

ROOT_DIR = os.path.abspath("../../")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

class CocoConfig(Config):
    NAME = "coco"
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

config = CocoConfig()
model_dir = os.path.join(ROOT_DIR, "logs")
model = modellib.MaskRCNN(mode="inference", model_dir=model_dir, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)