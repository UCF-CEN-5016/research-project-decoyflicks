import tensorflow as tf
from object_detection import export_saved_model_tpu_lib

batch_size = 1
height, width = 640, 640
input_data = tf.random.uniform((batch_size, height, width, 3), minval=0, maxval=255, dtype=tf.int32)

model_path = 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8'
config = export_saved_model_tpu_lib.TrainEvalPipelineConfig()
with tf.Session() as sess:
    restore_fn = tf.train.latest_checkpoint(model_path)
    sess.run(tf.global_variables_initializer())
    model_builder = export_saved_model_tpu_lib.create_model_from_config(config.model)
    graph_def, output_tensors = model_builder.build_graph(use_bfloat16=False)
    input_placeholder = tf.placeholder(tf.float32, shape=(batch_size, height, width, 3))
    sess.run(tf.train.init_from_checkpoint(restore_fn, {var.op.name: var for var in tf.global_variables()}))
    result_tensor_dict = sess.run(output_tensors, feed_dict={input_placeholder: input_data})

assert hasattr(result_tensor_dict['outputs'], 'DetectionFromImageModule')