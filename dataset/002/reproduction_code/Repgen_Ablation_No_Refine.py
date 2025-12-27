import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from object_detection.builders import model_builder

def main():
    # Load the pipeline configuration file
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile('models/centernet_mobilenetv2fp/pipeline.config', 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

    # Build the model
    model_fn = model_builder.build(pipeline_config.model, is_training=True)

    # Create a dummy input tensor for demonstration purposes
    dummy_input = tf.random.normal([1, 300, 300, 3])

    # Create a TensorFlow session and run the model
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        output_tensor = sess.run(model_fn(dummy_input))

if __name__ == '__main__':
    main()