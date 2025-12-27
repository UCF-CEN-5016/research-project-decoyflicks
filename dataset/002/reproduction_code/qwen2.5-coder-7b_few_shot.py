import tensorflow as tf

def import_graph_matcher():
    """
    Attempt to import tensorflow.contrib.quantize.python.graph_matcher.
    Prints the result and returns the imported module or None.
    """
    try:
        from tensorflow.contrib.quantize.python import graph_matcher  # noqa: F401
        print("Import successful")
        return graph_matcher
    except ModuleNotFoundError:
        print("ModuleNotFoundError: graph_matcher is not available in TensorFlow 2.x")
        return None

graph_matcher = import_graph_matcher()