import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder

config_path = 'pipeline.config'
configs = config_util.get_configs_from_pipeline_file(config_path)
model_config = configs['model']
model = model_builder.build(model_config=model_config, is_training=False)

print("Model loaded successfully")