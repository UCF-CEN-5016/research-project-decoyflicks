import tensorflow as tf
from object_detection import exporter
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

def main():
    model_dir = 'models/centernet_mobilenetv2fp'
    pipeline_config_path = 'models/centernet_mobilenetv2fp/pipeline.config'
    
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

    class MockModel:
        def preprocess(self, inputs):
            return tf.identity(inputs), []

        def predict(self, preprocessed_inputs, true_image_shapes):
            return {'image': tf.zeros([1, 320, 320, 3])}

        def postprocess(self, prediction_dict, true_image_shapes):
            return {
                'detection_boxes': tf.zeros([1, 10, 4]),
                'detection_scores': tf.zeros([1, 10]),
                'detection_classes': tf.zeros([1, 10]),
                'num_detections': tf.zeros([1])
            }

    mock_model = MockModel()
    exporter.export_inference_graph(
        input_type='image_tensor',
        pipeline_config=pipeline_config,
        trained_checkpoint_prefix=model_dir,
        output_directory=model_dir
    )

if __name__ == '__main__':
    main()