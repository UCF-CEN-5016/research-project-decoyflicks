import tensorflow as tf
import keras.layers as KL
import keras.models as KM
import numpy as np
import mrcnn.model as modellib

MODEL_DIR = './logs'
COCO_MODEL_PATH = './mask_rcnn_coco.h5'

class Config:
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes
    IMAGES_PER_GPU = 1

config = Config()
GPU_COUNT = 1
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

try:
    model.load_weights(COCO_MODEL_PATH, by_name=True)
except NotImplementedError as e:
    print(e)

dummy_image = np.random.rand(1, 1024, 1024, 3).astype(np.float32)

try:
    results = model.detect([dummy_image], verbose=1)
except NotImplementedError as e:
    print(e)