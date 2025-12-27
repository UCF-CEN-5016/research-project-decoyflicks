import tensorflow as tf
from google.protobuf import text_format

proto_str = b'\xe9' * 108 + b'config'

with tf.io.TFRecordWriter('test.tfrecord') as writer:
    writer.write(proto_str)

proto_config = tf.train.PipelineConfig()
proto_config.ParseFromString(proto_str)

print(proto_config)