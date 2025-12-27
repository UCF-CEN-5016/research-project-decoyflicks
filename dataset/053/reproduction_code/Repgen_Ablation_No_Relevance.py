import tensorflow as tf
from keras_cv import models, metrics, visualization

# Load COCO dataset using KerasCV's DataLoader (Assuming DataLoader is already defined and available)
dataset = models.COCODataLoader()

# Preprocess the dataset by resizing images to 640x640 pixels and normalizing pixel values (Assuming preprocessing function is available)
train_dataset, val_dataset = tf.keras.utils.split_dataset(dataset, 0.7)

# Create a YOLOV8 backbone model using KerasCV's YOLOV8Backbone.from_preset with 'yolo_v8_s_backbone_coco' preset
backbone = models.YOLOV8Backbone.from_preset("yolo_v8_s_backbone_coco")

# Initialize the YOLOV8Detector model with 10 classes (assuming class_mapping has 10 entries), bounding box format as 'xyxy', and fpn_depth as 1
yolo = models.YOLOV8Detector(num_classes=10, bounding_box_format="xyxy", backbone=backbone, fpn_depth=1)

# Compile the model using Adam optimizer with learning rate of 0.001 and global_clipnorm set to 5.0, classification loss as binary_crossentropy and box_loss as ciou
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, global_clipnorm=5.0)
yolo.compile(optimizer=optimizer, classification_loss="binary_crossentropy", box_loss="ciou")

# Prepare a validation dataset by splitting the original dataset into a training set of 70% and a validation set of 30%
val_dataset = tf.keras.utils.split_dataset(dataset, 0.3)

# Create an instance of EvaluateCOCOMetricsCallback with the validation dataset and 'model.h5' as save path
evaluate_callback = metrics.EvaluateCOCOMetricsCallback(val_dataset, "model.h5")

# Train the model for 3 epochs using fit method, passing the training and validation datasets and the callback
yolo.fit(train_dataset, validation_data=val_dataset, epochs=3, callbacks=[evaluate_callback])

# After training, assert that the best mAP score is greater than -1.0 to verify the callback functionality
assert evaluate_callback.best_map > -1.0

# Visualize detections on a subset of the validation dataset by calling visualize_detections function with the trained model, validation dataset and bounding box format 'xyxy'
visualization.visualize_detections(yolo, dataset=val_dataset, bounding_box_format="xyxy")

# Monitor and print the model's summary for verification purposes
yolo.summary()