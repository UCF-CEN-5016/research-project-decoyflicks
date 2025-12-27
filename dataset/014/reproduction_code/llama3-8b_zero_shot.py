import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format 

configpath = 'path_to_your_pipeline.config'

config = config_util.get_configs_from_pipeline_file(configpath)

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(configpath, "r") as f:                                                                                                                                                                                                                     
    proto_str = f.read()                                                                                                                                                                                                                                          
    text_format.Merge(proto_str, pipeline_config)  

try:
    pipeline_config.eval_input_reader.max_number_of_boxes = 500
except AttributeError:
    print("Bug reproduced")