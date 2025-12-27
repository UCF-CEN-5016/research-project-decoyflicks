import tensorflow as tf
   print(tf.__version__)  # Outputs: 2.13.0

import object_detection
   model = object_detection.model faster_rcnn_inception_resnet_v2_640x640

from tensorflow.keras.layers import Layer
   # Use compatible layers if any are deprecated in TF2.13.x