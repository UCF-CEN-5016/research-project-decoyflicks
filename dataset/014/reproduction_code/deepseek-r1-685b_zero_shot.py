import tensorflow as tf
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

config_content = """
model {
  faster_rcnn {
    num_classes: 1
    image_resizer {
      fixed_shape_resizer {
        height: 300
        width: 300
      }
    }
  }
}
train_config {
  batch_size: 1
}
eval_config {
  metrics_set: "coco_detection_metrics"
}
train_input_reader {
  label_map_path: "path/to/labelmap"
  tf_record_input_reader {
    input_path: "path/to/train.record"
  }
}
eval_input_reader {
  label_map_path: "path/to/labelmap"
  tf_record_input_reader {
    input_path: "path/to/val.record"
  }
}
"""

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
text_format.Parse(config_content, pipeline_config)
pipeline_config.eval_input_reader.max_number_of_boxes = 500