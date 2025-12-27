import tensorflow as tf
from official.vision.beta.tasks import image_classification

# Minimal setup for reproduction
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
else:
    print("No GPU found. Running on CPU")

# Define the model and dataset
model = image_classification.ImageClassificationTask(
    backbone='resnet_rs50',
    num_classes=1001,
    input_size=[160, 160, 3],
    train_data={
        'input_path': '/ppusw/datasets/vision/imagenet/tfrecords/train*',
        'global_batch_size': 2
    },
    validation_data={
        'input_path': '/ppusw/datasets/vision/imagenet/tfrecords/valid*',
        'global_batch_size': 2
    }
)

# Define the training loop
def train_loop(model, steps):
    for _ in range(steps):
        with tf.GradientTape() as tape:
            loss = model.train_step()
            if tf.math.is_nan(loss):
                print("Loss value is NaN after training loop")
                break

# Trigger the bug
train_loop(model, 100)