import tensorflow as tf
from object_detection import exporter_lib_v2
from object_detection.builders import model_builder
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

def reproduce_bug():
    # 1. Basic configuration (mimicking a trained model setup)
    config = """
    model {
      faster_rcnn {
        num_classes: 90
        image_resizer {
          fixed_shape_resizer {
            height: 640
            width: 640
          }
        }
        feature_extractor {
          type: 'faster_rcnn_resnet50_v1_640x640'
        }
      }
    }
    train_config {
      batch_size: 1
    }
    """
    
    # 2. Parse config
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    text_format.Merge(config, pipeline_config)
    
    # 3. Build model (mimicking trained state)
    model = model_builder.build(pipeline_config.model, is_training=False)
    fake_checkpoint = tf.train.Checkpoint(model=model)
    
    # 4. Try exporting - this should trigger the error
    try:
        exporter_lib_v2.export_inference_graph(
            input_type='image_tensor',
            pipeline_config=pipeline_config,
            trained_checkpoint_dir='/tmp/fake_ckpt',  # would fail anyway
            output_directory='/tmp/bug_output'
        )
    except AttributeError as e:
        print(f"Reproduced error: {e}")
        print("This occurs because DetectionFromImageModule isn't properly initialized")
        print("Potential fixes:")
        print("1. Upgrade to TF 2.5+ where this was fixed")
        print("2. Manually add outputs attribute to the module")
        print("3. Check your pipeline.config for compatibility issues")

if __name__ == '__main__':
    reproduce_bug()

def _add_output_to_module(module):
    if not hasattr(module, 'outputs'):
        module.outputs = module.model.postprocess(module.model(module.inputs))