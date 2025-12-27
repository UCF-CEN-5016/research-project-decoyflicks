import tensorflow as tf

def main():
    pipeline_config_path = 'ssd_efficientdet_d0_512x512_coco17_tpu-8.config'
    model_dir = 'training'

    with open(pipeline_config_path, 'rb') as f:
        proto_str = f.read()

    try:
        tf.compat.v1.app.run()
    except UnicodeDecodeError as e:
        print(e)

if __name__ == '__main__':
    main()