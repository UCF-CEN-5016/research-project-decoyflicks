import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.saved_model import tag_constants
from pathlib import Path
from tensorflow.keras.utils import get_file

def load_model(model_name):
    # Model download setup
    base_url = 'http://download.tensorflow.org/models/research/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True)
    
    # Model directory setup
    model_dir = Path(model_dir) / "saved_model"
    print(f"Model directory: {model_dir}")
    
    # Load model
    with tf.Session() as sess:
        tf.saved_model.loader.load(
            sess,
            [tag_constants.SERVING],
            str(model_dir))
        
        # Get tensors
        graph = tf.get_default_graph()
        tensor_dict = {
            'detection_boxes': graph.get_tensor_by_name('detection_boxes:0'),
            'detection_scores': graph.get_tensor_by_name('detection_scores:0'),
            'detection_classes': graph.get_tensor_by_name('detection_classes:0'),
            'num_detections': graph.get_tensor_by_name('num_detections:0')
        }
        
    return tensor_dict

# Test with a sample model (SSD MobileNet V2)
try:
    detection_model = load_model("ssd_mobilenet_v2_coco_2018_03_29")
    print("Model tensors:", detection_model)
except Exception as e:
    print("Error loading model:", e)