# Step 1: Necessary imports
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder

# Step 2: Verify environment
print("TF Version:", tf.__version__)  # Should be 2.13.0
print("Python Version:", sys.version)  # Should be 3.10.x

# Step 3: Model configuration (minimal setup)
pipeline_config = """
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
      type: "faster_rcnn_inception_resnet_v2_keras"
    }
  }
}
"""

# Step 4: Trigger the bug
configs = config_util.get_configs_from_pipeline_string(pipeline_config)
model_config = configs['model']
try:
    model = model_builder.build(model_config=model_config, is_training=False)
    print("Model built successfully")
except Exception as e:
    print("Error occurred:", str(e))