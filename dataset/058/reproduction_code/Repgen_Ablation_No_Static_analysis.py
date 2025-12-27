import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow_datasets as tfds

def preprocess_data(sample):
    image = sample["image"]
    label = sample["objects"]["label"]
    bbox = sample["objects"]["bbox"]
    
    # Resize and pad the image
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    image = preprocess_input(image)
    
    # Encode labels
    label = label_encoder.encode(label.numpy())
    return image, label, ratio

def visualize_detections(image, bbox, class_names, scores):
    fig, ax = plt.subplots(1)
    ax.imshow(image.numpy().astype(np.uint8))
    for i in range(len(bbox)):
        rect = Rectangle((bbox[i][0], bbox[i][1]), bbox[i][2] - bbox[i][0], bbox[i][3] - bbox[i][1], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(bbox[i][0], bbox[i][1], f"{class_names[i]} {scores[i]:.2f}", fontsize=12, color="red")
    plt.axis('off')
    plt.show()

def prepare_image(image):
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    image = preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio

(train_dataset, val_dataset), dataset_info = tfds.load("coco/2017", split=["train", "validation"], with_info=True, data_dir="data")

# Load the COCO2017 dataset using TensorFlow Datasets
autotune = tf.data.AUTOTUNE
train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)
train_dataset = train_dataset.shuffle(8 * batch_size)
train_dataset = train_dataset.padded_batch(batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True)
train_dataset = train_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
train_dataset = train_dataset.prefetch(autotune)

val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
val_dataset = val_dataset.padded_batch(batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True)
val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
val_dataset = val_dataset.prefetch(autotune)

# Initialize and compile model
resnet50_backbone = get_backbone()
loss_fn = RetinaNetLoss(num_classes)
model = RetinaNet(num_classes, resnet50_backbone)
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate_fn, momentum=0.9)
model.compile(loss=loss_fn, optimizer=optimizer)

# Load the latest weights from the specified directory
weights_dir = "data"
latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
model.load_weights(latest_checkpoint)

# Build an inference model
image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)
detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)

# Iterate over 2 samples from the validation dataset
for sample in val_dataset.take(2):
    image = tf.cast(sample["image"], dtype=tf.float32)
    input_image, ratio = prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    class_names = [int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]]
    visualize_detections(image, detections.nmsed_boxes[0][:num_detections] / ratio, class_names, detections.nmsed_scores[0][:num_detections])