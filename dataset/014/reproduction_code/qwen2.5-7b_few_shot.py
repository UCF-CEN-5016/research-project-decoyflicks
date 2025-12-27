import pipeline_pb2  # Assuming this is the protobuf module for the object detection pipeline

# Initialize the config
config = pipeline_pb2.TrainEvalPipelineConfig()

# Attempt to set a field on the repeated container (invalid)
config.eval_input_reader.CopyFrom(pipeline_pb2.InputReader(max_number_of_boxes=500))