import os
import tensorflow as tf
from tensorflow import keras
from keras_cv.models.object_detection import RetinaNet, LabelEncoder

# Assuming these functions are defined elsewhere in the codebase
# from your_module import swap_xy, random_flip_horizontal, resize_and_pad_image, convert_to_xywh, get_backbone, RetinaNetLoss

tf.random.set_seed(42)

batch_size = 2
num_classes = 80

def preprocess_data(sample):
    image = sample["image"]
    bbox = swap_xy(sample["objects"]["bbox"])  # Ensure this function is defined
    class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)
    image, bbox = random_flip_horizontal(image, bbox)  # Ensure this function is defined
    image, image_shape, _ = resize_and_pad_image(image)  # Ensure this function is defined
    bbox = tf.stack([
        bbox[:, 0] * image_shape[1],
        bbox[:, 1] * image_shape[0],
        bbox[:, 2] * image_shape[1],
        bbox[:, 3] * image_shape[0],
    ], axis=-1)
    bbox = convert_to_xywh(bbox)  # Ensure this function is defined
    return image, bbox, class_id

model_dir = 'retinanet/'
label_encoder = LabelEncoder()

# Load your custom dataset here
train_dataset = ...  
# Load your validation dataset here
val_dataset = ...    

train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.padded_batch(batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True)
train_dataset = train_dataset.map(label_encoder.encode_batch, num_parallel_calls=tf.data.AUTOTUNE)

val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.padded_batch(batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True)
val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=tf.data.AUTOTUNE)

resnet50_backbone = get_backbone()  # Ensure this function is defined
model = RetinaNet(num_classes, resnet50_backbone)
model.compile(loss=RetinaNetLoss(num_classes), optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9))  # Ensure RetinaNetLoss is defined

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, 'weights_epoch_{epoch}'), 
        save_best_only=False
    )
]

# The following line is crucial for reproducing the bug
model.fit(
    train_dataset.take(20),
    validation_data=val_dataset.take(20),
    epochs=1,
    callbacks=callbacks_list,
)