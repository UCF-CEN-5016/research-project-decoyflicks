import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D

config = None
MODEL_DIR = 'path_to_model_dir'
COCO_MODEL_PATH = 'path_to_coco_model'

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Additional imports for the script to run without error
import modellib