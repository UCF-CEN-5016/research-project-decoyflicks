import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder

pipeline_config = tf.io.gfile.GFile(
    'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8/pipeline.config', 'r').read()

with open('pipeline.config', 'w') as f:
    f.write(pipeline_config)

configs = config_util.get_configs_from_pipeline_file('pipeline.config')
model_config = configs['model']
model = model_builder.build(model_config=model_config, is_training=False)