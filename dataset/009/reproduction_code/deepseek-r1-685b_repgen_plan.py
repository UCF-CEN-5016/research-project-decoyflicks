import tensorflow as tf
from object_detection import exporter_lib_v2
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

def create_pipeline_config():
    config_text = """
    model {
      faster_rcnn {
        num_classes: 1
        image_resizer {
          fixed_shape_resizer {
            height: 640
            width: 640
          }
        }
        feature_extractor {
          type: "faster_rcnn_resnet50_v1_640x640_coco17_tpu-8"
        }
      }
    }
    train_config {
      batch_size: 1
    }
    """
    
    config = pipeline_pb2.TrainEvalPipelineConfig()
    text_format.Merge(config_text, config)
    
    return config

def simulate_trained_checkpoint(checkpoint_path):
    tf.io.gfile.makedirs(checkpoint_path)
    tf.train.Checkpoint().save(checkpoint_path + "/ckpt-0")

def export_inference_graph(config, checkpoint_path, output_directory):
    try:
        exporter_lib_v2.export_inference_graph(
            input_type="image_tensor",
            pipeline_config=config,
            trained_checkpoint_dir=checkpoint_path,
            output_directory=output_directory
        )
    except AttributeError as e:
        print(f"Error encountered: {e}")
        print("This reproduces the 'DetectionFromImageModule' missing 'outputs' attribute error")

if __name__ == "__main__":
    config = create_pipeline_config()
    checkpoint_path = "/tmp/dummy_ckpt"
    simulate_trained_checkpoint(checkpoint_path)
    export_inference_graph(config, checkpoint_path, "/tmp/exported_model")