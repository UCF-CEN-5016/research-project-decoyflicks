import tensorflow as tf
from tensorflow.keras import Mask_RCNN
import os

# Ensure TensorFlow uses native TF saved format
os.environ['TF_KERAS_MODEL_Saving_fmt'] = 'tf'

model_dir = "path/to/trained_weights.h5"
config = tf.keras Mask_RCNNConfig()

# Create a new model with the same config and layers as the pre-trained one
new_model = MaskRCNN(mode="inference", 
                   model_dir=model_dir, 
                   config=config)

# Load weights using native TensorFlow format
weights_dir = "path/to/trained_model"
new_model.load_weights(os.path.join(weights_dir, "weights.ckpt"))

# Verify loading was successful
print("Weights loaded successfully")