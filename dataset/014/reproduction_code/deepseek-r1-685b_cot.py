import tensorflow as tf
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# Create minimal config content that reproduces the issue
config_content = """
model {
  faster_rcnn {
    num_classes: 1
    first_stage_max_proposals: 100
    second_stage_post_processing {
      batch_non_max_suppression {
        max_total_detections: 100
        max_detections_per_class: 100
      }
    }
  }
}
train_config {
  max_number_of_boxes: 100
}
eval_config {
  num_visualizations: 100
  max_num_boxes_to_visualize: 100
}
eval_input_reader {
  shuffle: false
  num_readers: 1
}
"""

def reproduce_bug():
    # Create config proto
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    text_format.Merge(config_content, pipeline_config)
    
    # This will fail - trying to set attribute on repeated field container
    try:
        pipeline_config.eval_input_reader.max_number_of_boxes = 500
    except AttributeError as e:
        print(f"Expected error when wrong access: {e}")
    
    # Correct way - access first reader in repeated field
    if len(pipeline_config.eval_input_reader) > 0:
        pipeline_config.eval_input_reader[0].max_number_of_boxes = 500
        print("Successfully set max_number_of_boxes on first eval_input_reader")
    
    # Show other required parameters that need updating
    required_updates = {
        'train_config.max_number_of_boxes': 500,
        'eval_config.max_num_boxes_to_visualize': 500,
        'eval_config.num_visualizations': 500,
        'model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.max_detections_per_class': 500,
        'model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.max_total_detections': 500,
        'model.faster_rcnn.first_stage_max_proposals': 500
    }
    
    print("\nOther parameters that need updating for >100 objects:")
    for param, value in required_updates.items():
        print(f"{param}: {value}")

if __name__ == "__main__":
    reproduce_bug()