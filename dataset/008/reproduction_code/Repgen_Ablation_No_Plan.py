import tensorflow as tf
from object_detection.utils import config_util
from official.vision.detection.modeling import factory

PATH_TO_CFG = "/tmp/models/faster_rcnn_inception_resnet_v2_640x640/pipeline.config"
config = tf.compat.v1.estimator.run_config.RuntimeConfig(model_dir="/tmp/models/faster_rcnn_inception_resnet_v2_640x640/")
model_config = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)

from official.vision.detection import hyperparams_config
model = factory.build_model(model_config=model_config.model, is_training=False, add_summaries=True)