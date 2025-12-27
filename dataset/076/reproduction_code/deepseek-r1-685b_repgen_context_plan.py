import os
from mrcnn import model as modellib
from mrcnn.config import Config

class TestConfig(Config):
    NAME = "test"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 81  # COCO classes

def setup_model():
    # Setup paths and config
    MODEL_DIR = os.path.expanduser("~/models")
    COCO_MODEL_PATH = os.path.expanduser("~/mask_rcnn_coco.h5")  # Must exist
    config = TestConfig()
    
    # This will raise NotImplementedError in TF 2.5+ with HDF5 weights
    model = modellib.MaskRCNN(mode="inference", 
                              model_dir=MODEL_DIR, 
                              config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)  # Fails here
    return model

if __name__ == "__main__":
    trained_model = setup_model()