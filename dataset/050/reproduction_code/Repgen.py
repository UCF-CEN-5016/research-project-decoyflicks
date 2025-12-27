import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow_datasets import load

# Set up batch size and image dimensions
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

# Load Oxford Flowers dataset
train_dataset, test_dataset = load('oxford_flowers102', split=['train[:80%]', 'train[80%:]'], shuffle_files=True, with_info=False, batch_size=BATCH_SIZE)

# Preprocess the dataset
def preprocess_for_model(inputs):
    images, labels = inputs["images"], inputs["labels"]
    images = tf.cast(images, tf.float32)
    return images, labels

train_dataset = train_dataset.map(preprocess_for_model)
test_dataset = test_dataset.map(preprocess_for_model)

# Construct a custom model based on EfficientNetV2S from KerasCV
def get_model():
    input_shape = IMAGE_SIZE + (3,)
    model = models.ImageClassifier.from_preset(
        "efficientnetv2_s", num_classes=102
    )
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        optimizer=tf.keras.optimizers.SGD(momentum=0.9),
        metrics=["accuracy"],
    )
    return model

model = get_model()

# Train the model for 1 epoch
model.fit(
    train_dataset,
    epochs=1,
    validation_data=test_dataset,
)