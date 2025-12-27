import os
import tensorflow as tf
from tensorflow import keras
import keras_cv
from keras_cv import bounding_box
from keras_cv import visualization
import xml.etree.ElementTree as ET  # Added import for ET
from tqdm import tqdm  # Added import for tqdm

SPLIT_RATIO = 0.2
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCH = 5
GLOBAL_CLIPNORM = 1.0  # Added definition for GLOBAL_CLIPNORM

path_images = '/kaggle/input/dataset/data/images/'
path_annot = '/kaggle/input/dataset/data/annotations/'

xml_files = sorted([os.path.join(path_annot, file_name) for file_name in os.listdir(path_annot) if file_name.endswith('.xml')])
jpg_files = sorted([os.path.join(path_images, file_name) for file_name in os.listdir(path_images) if file_name.endswith('.jpg')])

# Placeholder for class_mapping, should be defined based on the dataset
class_mapping = {}  # Added definition for class_mapping

def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    image_name = root.find("filename").text
    image_path = os.path.join(path_images, image_name)
    boxes = []
    classes = []
    for obj in root.iter("object"):
        cls = obj.find("name").text
        classes.append(cls)
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
    class_ids = [list(class_mapping.keys())[list(class_mapping.values()).index(cls)] for cls in classes]
    return image_path, boxes, class_ids

image_paths = []
bbox = []
classes = []
for xml_file in tqdm(xml_files):
    image_path, boxes, class_ids = parse_annotation(xml_file)
    image_paths.append(image_path)
    bbox.append(boxes)
    classes.append(class_ids)

bbox = tf.ragged.constant(bbox)
classes = tf.ragged.constant(classes)
image_paths = tf.ragged.constant(image_paths)

data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))

num_val = int(len(xml_files) * SPLIT_RATIO)
val_data = data.take(num_val)
train_data = data.skip(num_val)

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

def load_dataset(image_path, classes, bbox):
    image = load_image(image_path)
    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": bbox,
    }
    return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}

train_ds = train_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(BATCH_SIZE * 4)
train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)

val_ds = val_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.shuffle(BATCH_SIZE * 4)
val_ds = val_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)

backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_s_backbone_coco")

yolo = keras_cv.models.YOLOV8Detector(
    num_classes=len(class_mapping),
    bounding_box_format="xyxy",
    backbone=backbone,
    fpn_depth=1,
)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    global_clipnorm=GLOBAL_CLIPNORM,
)

yolo.compile(optimizer=optimizer, classification_loss="binary_crossentropy", box_loss="ciou")

yolo.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,
)

# Potential typo check
level2_dau_2 = None  # Declared but not used
# Function call that may be incorrect
selective_kernel_feature_fusion(level2_dau_2)  # Changed to use level2_dau_2 instead of level3_dau_2