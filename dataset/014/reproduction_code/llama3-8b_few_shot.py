import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

configpath = 'path/to/pipeline.config'  # Replace with your actual file path

config = config_util.get_configs_from_pipeline_file(configpath)
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(configpath, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

# Trigger the bug by setting max_number_of_boxes to 100 (default value)
# and observing the model's behavior
pipeline_config.eval_input_reader.max_number_of_boxes = 100

print("Model configuration:")
print(config_util.make_proto_config(pipeline_config))

# Simulate training or evaluation here to reproduce the issue
# For example:
for _ in range(270000):  # Train for 270K steps
    pass

print("Accuracy:", tf.keras.metrics.Accuracy().get().numpy())  # Expected: <0.04 due to bug