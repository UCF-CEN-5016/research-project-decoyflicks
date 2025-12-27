import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Force TF1 compatibility mode
import pathlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# TF1-compatible model loading function
def load_model_tf1(model_name):
    base_url = 'http://download.tensorflow.org/models/research/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True)
    
    model_dir = pathlib.Path(model_dir)
    
    # TF1 way to load frozen graph
    model_path = str(model_dir / 'frozen_inference_graph.pb')
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    return detection_graph

# Test with a TF1-compatible model (e.g., ssd_mobilenet_v1_coco_2018_01_28)
model_name = 'ssd_mobilenet_v1_coco_2018_01_28'
detection_graph = load_model_tf1(model_name)

# Verify the tensors
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Get tensor names
        ops = detection_graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        
        # Print some tensor info
        for tensor_name in all_tensor_names:
            if 'detection' in tensor_name:
                print(tensor_name)
                
        # Should show proper tensor names without "???"