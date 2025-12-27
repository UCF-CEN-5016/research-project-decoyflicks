import os
from mrcnn import model as modellib
from mrcnn.config import Config

class TestConfig(Config):
    NAME = "test"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 81  # COCO classes

def setup_model():
    MODEL_DIR = os.path.expanduser("~/models")
    COCO_MODEL_PATH = os.path.expanduser("~/mask_rcnn_coco.h5")  # Must exist
    config = TestConfig()
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    return model, COCO_MODEL_PATH

def load_model_weights(model, weights_path):
    model.load_weights(weights_path, by_name=True)  # Fails here

if __name__ == "__main__":
    model, weights_path = setup_model()
    load_model_weights(model, weights_path)