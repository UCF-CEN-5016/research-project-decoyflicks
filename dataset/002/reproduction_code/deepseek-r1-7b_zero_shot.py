import tensorflow as tf

# Make sure you have installed the required TensorFlow contrib packages:
# For example, if using Tensorflow 2.x with Object Detection models
from tensorflow.contrib import graph_matcher
from tensorflow.contrib import quantize

# Minimal code to test (adjust according to your model path)
export_dir = "path/to/exported/graph.pb"
input_saved_model_dir = "path/to/exported/saved_model"
output_dir = "path/to/output"

model = tf.saved_model.load(input_saved_model_dir, signature_name=None, tags=None)
converted_graph_def = graph_matcher.match_graph_def(model.graph.as_graph_def(), 
                                                  quantize.QUANTIZEConfig(
                                                      quantize.QuantizationMode.FULLY
                                                      QUANTIZE_AWARE Training=False))
tf.train.write_graph(converted_graph_def, export_dir, output_dir + '/frozen.pb', as_text=False)