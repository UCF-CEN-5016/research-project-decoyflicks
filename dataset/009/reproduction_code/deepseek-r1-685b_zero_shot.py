import tensorflow as tf
from object_detection import exporter_lib_v2
from object_detection.builders import model_builder
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

def build_fake_model_config():
    config = pipeline_pb2.TrainEvalPipelineConfig()
    text_format.Merge("""
    model {
        faster_rcnn {
            num_classes: 1
            image_resizer {
                fixed_shape_resizer {
                    height: 640
                    width: 640
                }
            }
        }
    }
    """, config)
    return config

class FakeDetectionModel(tf.keras.Model):
    def __init__(self):
        super(FakeDetectionModel, self).__init__()
        self.dummy = tf.keras.layers.Dense(1)

    def call(self, inputs):
        return {'detection_boxes': tf.zeros([1, 10, 4]), 
                'detection_classes': tf.zeros([1, 10]),
                'detection_scores': tf.zeros([1, 10])}

def build_fake_model():
    return FakeDetectionModel()

def reproduce_bug():
    config = build_fake_model_config()
    model = build_fake_model()
    checkpoint_path = '/tmp/fake_checkpoint'
    model.save_weights(checkpoint_path)
    
    exporter_lib_v2.export_inference_graph(
        input_type='image_tensor',
        pipeline_config=config,
        trained_checkpoint_dir=checkpoint_path,
        output_directory='/tmp/exported_model'
    )

reproduce_bug()