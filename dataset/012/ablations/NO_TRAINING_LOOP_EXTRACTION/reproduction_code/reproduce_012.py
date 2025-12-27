import tensorflow as tf
from official.modeling.optimization.optimizer_factory import OptimizerFactory
from official.modeling.optimization import ema_optimizer
import argparse

# Set TensorFlow version
tf.__version__ = 'tf-nightly-cpu 2.16.0.dev20231103'
tf_keras = tf.keras

# Define dataset directory
data_dir = '/dir_to_imagenet_data'

# Command-line argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train_and_eval')
parser.add_argument('--model_type', type=str, default='resnet')
parser.add_argument('--dataset', type=str, default='imagenet')
parser.add_argument('--data_dir', type=str, default=data_dir)
args = parser.parse_args()

# Attempt to access the Optimizer class
try:
    optimizer = ema_optimizer.ExponentialMovingAverage(tf_keras.optimizers.legacy.Optimizer)
except AttributeError as e:
    print(e)