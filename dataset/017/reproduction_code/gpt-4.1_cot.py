import os
import pathlib
import sys
import tarfile
import urllib.request
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Check TensorFlow version (should be 1.x)
print("TensorFlow version:", tf.__version__)
assert tf.__version__.startswith('1'), "This script is for TensorFlow 1.x"

# Download and extract model
def download_model(model_name, base_url):
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_file,
                                        untar=True)
    return pathlib.Path(model_dir)

# Load frozen graph into TF1 graph
def load_frozen_graph(frozen_graph_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(str(frozen_graph_path), "rb") as f:
            serialized_graph = f.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name="")
    return detection_graph

# Load image into numpy array
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# Visualization function to show boxes on image
def visualize_boxes_and_labels(image_np, boxes, classes, scores, category_index, min_score_thresh=0.5):
    plt.figure(figsize=(12,8))
    plt.imshow(image_np)
    ax = plt.gca()

    im_height, im_width = image_np.shape[:2]

    # Draw boxes
    for i in range(boxes.shape[0]):
        if scores[i] > min_score_thresh:
            box = boxes[i]
            ymin, xmin, ymax, xmax = box
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
            rect = patches.Rectangle((left, top), right - left, bottom - top,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            class_name = category_index.get(classes[i], 'N/A')
            ax.text(left, top - 10, '{}: {:.2f}'.format(class_name, scores[i]),
                    color='red', fontsize=12, backgroundcolor='white')
    plt.axis('off')
    plt.show()

# Main function
def main():
    # Model to download (TF1 model)
    MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
    BASE_URL = 'http://download.tensorflow.org/models/object_detection/'

    print("Downloading model...")
    model_dir = download_model(MODEL_NAME, BASE_URL)
    print("Model downloaded to:", model_dir)

    frozen_graph_path = model_dir / 'frozen_inference_graph.pb'
    assert frozen_graph_path.exists(), "Frozen graph file not found!"

    print("Loading frozen graph...")
    detection_graph = load_frozen_graph(frozen_graph_path)

    # Load label map for COCO (simple dict for demo)
    category_index = {
        1: 'person',
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        10: 'traffic light',
        # Add more classes as needed...
    }

    # Load test image
    IMAGE_PATH = 'test_image.jpg'  # Provide your test image path here
    if not os.path.exists(IMAGE_PATH):
        # Download a sample image if not present
        img_url = 'https://upload.wikimedia.org/wikipedia/commons/6/60/Trafalgar_Square%2C_London_2_-_Jun_2009.jpg'
        urllib.request.urlretrieve(img_url, IMAGE_PATH)
        print("Sample image downloaded.")

    image = Image.open(IMAGE_PATH)
    image_np = load_image_into_numpy_array(image)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Get handles to input and output tensors
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
            classes_tensor = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections_tensor = detection_graph.get_tensor_by_name('num_detections:0')

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # Run inference
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes_tensor, scores_tensor, classes_tensor, num_detections_tensor],
                feed_dict={image_tensor: image_np_expanded})

            # Squeeze outputs to proper shape
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)

            # Visualize detection results
            visualize_boxes_and_labels(image_np, boxes, classes, scores, category_index)

if __name__ == '__main__':
    main()