import tensorflow as tf
from google.protobuf import text_format

OUTPUT_TFRECORD = 'test.tfrecord'
RAW_PROTO_BYTES = b'\xe9' * 108 + b'config'


def write_tfrecord(path: str, data: bytes) -> None:
    with tf.io.TFRecordWriter(path) as writer:
        writer.write(data)


def parse_pipeline_config_from_bytes(data: bytes) -> tf.train.PipelineConfig:
    pipeline_config = tf.train.PipelineConfig()
    pipeline_config.ParseFromString(data)
    return pipeline_config


def main():
    write_tfrecord(OUTPUT_TFRECORD, RAW_PROTO_BYTES)
    pipeline_config = parse_pipeline_config_from_bytes(RAW_PROTO_BYTES)
    print(pipeline_config)


if __name__ == "__main__":
    main()