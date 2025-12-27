import pipeline_pb2  # Assuming this is the protobuf module for the object detection pipeline

# Initialize the config
config = pipeline_pb2.TrainEvalPipelineConfig()

# Incorrectly attempt to set a field directly on a repeated container (will raise AttributeError)
config.eval_input_reader.max_number_of_boxes = 500