import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder

# This example assumes you have the pipeline config and checkpoint downloaded from the TF2 detection model zoo
# For minimal reproduction, simulate loading a TF1 checkpoint with TF2 model API which causes the error.

pipeline_config_path = 'faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8/pipeline.config'
checkpoint_path = 'faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8/checkpoint/ckpt-0'

# Load pipeline config
configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
model_config = configs['model']

# Build the model (TF2 way)
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Try to restore TF1 checkpoint (this triggers ValueError)
ckpt = tf.train.Checkpoint(model=detection_model)
ckpt.restore(checkpoint_path).expect_partial()

# Dummy input to run detection
@tf.function
def detect(input_tensor):
    return detection_model(input_tensor, training=False)

dummy_image = tf.zeros([1, 640, 640, 3], dtype=tf.float32)

# This line triggers the error related to unsupported ValueError for tf2 version
detections = detect(dummy_image)
print(detections)