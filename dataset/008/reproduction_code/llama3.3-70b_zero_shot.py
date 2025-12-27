import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.protos import model_pb2
from google.protobuf import text_format

# Load the pipeline configuration
pipeline_config = '''
model {
  faster_rcnn {
    num_classes: 90
    image_resizer {
      fixed_shape_resizer {
        height: 640
        width: 640
      }
    }
    feature_extractor {
      type: 'faster_rcnn_inception_resnet_v2'
    }
  }
}
'''

# Create a model builder
model_config = model_pb2.DetectionModel()
text_format.Merge(pipeline_config, model_config)
model_builder_func = model_builder.build

# Try to build the model
model = model_builder_func(model_config, is_training=False)

# Load the pre-trained model
pretrained_model = tf.saved_model.load('faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8/saved_model')

# Try to use the pre-trained model
pretrained_model.signatures['serving_default'](tf.random.normal([1, 640, 640, 3]))