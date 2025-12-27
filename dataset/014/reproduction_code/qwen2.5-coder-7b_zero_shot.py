import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


DEFAULT_PIPELINE_PATH = "path_to_your_pipeline.config"


def load_configs_from_pipeline_file(path):
    """
    Keep side effects identical to original: call config_util.get_configs_from_pipeline_file.
    Returns whatever that function returns (unused here) to preserve behavior.
    """
    return config_util.get_configs_from_pipeline_file(path)


def parse_pipeline_proto(path):
    """
    Read the pipeline config file and parse it into a TrainEvalPipelineConfig proto.
    """
    pipeline_cfg = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_cfg)
    return pipeline_cfg


def set_eval_max_number_of_boxes(pipeline_cfg, value):
    """
    Attempt to set eval_input_reader.max_number_of_boxes; on AttributeError print the same message.
    """
    try:
        pipeline_cfg.eval_input_reader.max_number_of_boxes = value
    except AttributeError:
        print("Bug reproduced")


def main(pipeline_path=DEFAULT_PIPELINE_PATH, max_boxes=500):
    # Preserve original call for side effects/behavior
    _ = load_configs_from_pipeline_file(pipeline_path)

    pipeline_cfg = parse_pipeline_proto(pipeline_path)
    set_eval_max_number_of_boxes(pipeline_cfg, max_boxes)


if __name__ == "__main__":
    main()