import tensorflow as tf

print("TF Version:", tf.__version__)  # Should be 2.13.0

# Attempt to load pretrained Faster RCNN model
model_url = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz"

try:
    # Load pretrained Faster RCNN model
    model = tf.keras.applications.InceptionResNetV2(
        weights='imagenet',
        include_top=False
    )
    print("Model loaded successfully")
except ValueError as e:
    print(f"Error loading model: {e}")
    print("This occurs because the model format is incompatible with TF2.13")