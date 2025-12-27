import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

def reproduce_bug():
    # Create a minimal config that would trigger the issue
    config_text = """
    model {
      faster_rcnn {
        num_classes: 1
        second_stage_post_processing {
          batch_non_max_suppression {
            max_detections_per_class: 100  # Default problematic value
            max_total_detections: 100
          }
        }
      }
    }
    train_config {
      max_number_of_boxes: 100
    }
    eval_config {
      max_num_boxes_to_visualize: 100
      num_visualizations: 100
    }
    eval_input_reader {
      # Missing max_number_of_boxes field
    }
    """
    
    # Parse the config
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    text_format.Parse(config_text, pipeline_config)
    
    # Attempt to set the missing field (will trigger the error)
    try:
        pipeline_config.eval_input_reader.max_number_of_boxes = 500
        print("Successfully set max_number_of_boxes")
    except AttributeError as e:
        print(f"Error encountered: {e}")
        print("This matches the reported bug behavior")

if __name__ == "__main__":
    reproduce_bug()