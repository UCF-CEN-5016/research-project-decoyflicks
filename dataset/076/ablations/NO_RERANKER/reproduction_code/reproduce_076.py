import tensorflow as tf
import keras.backend as K
import keras.layers as KL
import keras.models as KM
import numpy as np
import os
import mrcnn.model as modellib

MODEL_DIR = os.path.join(os.getcwd(), 'logs')
COCO_MODEL_PATH = os.path.join(MODEL_DIR, 'mask_rcnn_coco.h5')

class Config:
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes
    IMAGES_PER_GPU = 1

config = Config()
mode = "inference"
model = modellib.MaskRCNN(mode=mode, model_dir=MODEL_DIR, config=config)

model.load_weights(COCO_MODEL_PATH, by_name=True)