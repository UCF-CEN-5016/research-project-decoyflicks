import os
import numpy as np
import tensorflow as tf
import keras.backend as K
from mrcnn import model as modellib
from mrcnn.config import Config

class InferenceConfig(Config):
    NAME = "test"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    IMAGE_RESIZE_MODE = "square"
    IMAGE_SHAPE = [1024, 1024, 3]
    IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + NUM_CLASSES

config = InferenceConfig()

ROOT_DIR = os.path.abspath("../")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

try:
    model.load_weights(COCO_MODEL_PATH, by_name=True)
except NotImplementedError as e:
    print("Caught NotImplementedError:", e)