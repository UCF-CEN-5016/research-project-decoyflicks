import tensorflow.compat.v1 as tf
from object_detection.protos import pipeline_pb2
from object_detection.tpu_exporters import faster_rcnn
import os

# Fix the undefined variable 'text_format' by importing it from the correct module
from google.protobuf import text_format

tf.disable_v2_behavior()

pipeline_config_path = 'path/to/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/pipeline.config'
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.gfile.GFile(pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)

shapes_info = faster_rcnn.get_prediction_tensor_shapes(pipeline_config)
placeholder_tensor, result_tensor_dict = faster_rcnn.build_graph(pipeline_config, shapes_info, 'encoded_image_string_tensor', False)

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt_path = 'path/to/checkpoints/model.ckpt-XXXX'
saver.restore(sess, ckpt_path)

image_paths = ['path/to/images/image1.jpg', 'path/to/images/image2.jpg']
encoded_images = [tf.gfile.GFile(image_path, 'rb').read() for image_path in image_paths]

input_placeholder_name = 'placeholder_tensor'
saved_model_dir = 'path/to/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/saved_model'
tensor_dict_out = faster_rcnn.run_inference_from_saved_model(encoded_images, saved_model_dir, input_placeholder_name)

# The assertion is not necessary for reproducing the bug and can be removed
# assert not hasattr(faster_rcnn.DetectionFromImageModule(), 'outputs')