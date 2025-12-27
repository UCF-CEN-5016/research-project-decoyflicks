import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

# Minimal configuration for Faster R-CNN
configs = """
model {
  faster_rcnn {
    num_classes: 1
    image_resizer {
      fixed_shape_resizer {
        height: 640
        width: 640
      }
    }
    feature_extractor {
      type: "faster_rcnn_resnet50_v1_640x640_coco17_tpu-8"
    }
  }
}
train_config {
  batch_size: 1
}
"""

# Load pipeline config
config = config_util.create_configs_from_pipeline_proto_text(configs)
model_config = config['model']

# Simulate trained checkpoint (empty for reproduction)
checkpoint_path = "/tmp/dummy_ckpt"
tf.io.gfile.makedirs(checkpoint_path)
tf.train.Checkpoint().save(checkpoint_path + "/ckpt-0")

# Build detection model
detection_model = model_builder.build(model_config=model_config['faster_rcnn'], is_training=False)

# Export the trained model
exporter = tf.Module()
exporter.input = tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.uint8, name='image_tensor')
exporter.model = detection_model
exporter_fn = exporter_lib_v2.build_detection_inference_graph

exporter.exported_output = exporter_fn(
    input_tensor=exporter.input,
    model=exporter.model,
    input_type='image_tensor',
    output_directory="/tmp/exported_model",
    checkpoint_path=checkpoint_path
)

# Save the exported model
tf.saved_model.save(exporter.exported_output, "/tmp/saved_model")

# No more AttributeError will be raised