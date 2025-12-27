import tensorflow as tf
import tarfile
import requests

def download_model(model_url, model_path):
    response = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)

def extract_model(model_path, extract_path):
    with tarfile.open(model_path) as tar:
        tar.extractall(extract_path)

def load_model(model_dir):
    model = tf.saved_model.load(model_dir)
    return model

def main():
    model_url = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz"
    model_path = "faster_rcnn.tar.gz"
    extract_path = "faster_rcnn"

    download_model(model_url, model_path)
    extract_model(model_path, extract_path)
    
    try:
        model = load_model(extract_path)
        input_tensor = tf.random.normal([1, 640, 640, 3])
        detections = model(input_tensor)
        print(detections)
    except ValueError as e:
        print("Model loading or inference failed:", e)

if __name__ == "__main__":
    main()