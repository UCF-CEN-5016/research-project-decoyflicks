import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2

config_path = "path/to/your/pipeline.config"

pipeline_config = config_util.get_configs_from_pipeline_file(config_path)
pipeline_config = pipeline_config['model']

pipeline_config.eval_input_reader.max_number_of_boxes = 500