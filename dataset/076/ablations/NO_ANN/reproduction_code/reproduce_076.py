import tensorflow as tf
import keras.layers as KL
import keras.models as KM
import numpy as np
import mrcnn.model as modellib

MODEL_DIR = './logs'
COCO_MODEL_PATH = './mask_rcnn_coco.h5'

class Config:
    NUM_CLASSES = 1 + 80  # COCO has 80 classes
    IMAGES_PER_GPU = 1

config = Config()
GPU_COUNT = 1
model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)