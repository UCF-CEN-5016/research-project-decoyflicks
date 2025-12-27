import tensorflow as tf
from tensorflow.contrib.quantize import graph_matcher as graph_matcher_mod

def execute_graph_matcher():
    """Invoke the graph_matcher from TensorFlow contrib.quantize."""
    graph_matcher_mod.graph_matcher()

def main(argv=None):
    """Entry point for tf.app.run. The argv parameter is unused."""
    execute_graph_matcher()

if __name__ == '__main__':
    tf.app.run(main=main)