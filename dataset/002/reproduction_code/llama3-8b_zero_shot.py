import tensorflow as tf
from tensorflow.contrib.quantize import graph_matcher

def main(_):
    graph_matcher.graph_matcher()

if __name__ == '__main__':
    tf.app.run(main=main)