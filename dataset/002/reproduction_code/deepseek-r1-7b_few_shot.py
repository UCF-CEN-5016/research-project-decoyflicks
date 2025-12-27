# File: exporter.py
try:
    # Try importing deprecated TensorFlow.contrib.quantize.graph_matcher
    from tensorflow.contrib.quantize.python import graph_matcher
except ImportError:
    # If contrib is not available, handle the missing module by using alternative approach
    print("(tensorflow.contrib.quantize.graph_matcher) could not be imported. "
          "Switching to tf.lite based model export.")
    # Use TensorFlow Lite for model export instead of contrib
    from tensorflow.tflite import constants as tflite_constants
    from tensorflow.tflite import representative_data_gen

    def graph_matcher(input_graph_def, input_node_names):
        """Fallback implementation using tf.lite utilities"""
        return representative_data_gen.generate RepresentativeDataGen(
            input_graph_def,
            input_node_names,
            num_representative_data=100
        )