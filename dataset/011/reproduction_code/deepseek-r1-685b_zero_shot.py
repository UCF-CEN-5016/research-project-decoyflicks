import tensorflow as tf
from official.vision.beta.configs import image_classification
from official.vision.beta.tasks import image_classification as image_classification_task
from official.core import train_lib
from official.modeling import performance

# Setup config
config = image_classification.ImagenetResNetRSConfig()
config.task.model.backbone.resnet.model_id = 50
config.task.model.input_size = [160, 160, 3]
config.task.train_data.global_batch_size = 2
config.task.validation_data.global_batch_size = 2
config.task.train_data.dtype = 'float16'
config.task.validation_data.dtype = 'float16'
config.runtime.mixed_precision_dtype = 'float16'
config.trainer.train_steps = 100
config.trainer.optimizer_config.learning_rate.cosine.decay_steps = 100

# Create dummy dataset
def create_dummy_dataset(batch_size):
    images = tf.random.normal([batch_size, 160, 160, 3], dtype=tf.float16)
    labels = tf.random.uniform([batch_size], 0, 1000, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensors((images, labels))
    dataset = dataset.repeat()
    return dataset

# Setup task
task = image_classification_task.ImageClassificationTask(config.task)
train_ds = create_dummy_dataset(config.task.train_data.global_batch_size)
validation_ds = create_dummy_dataset(config.task.validation_data.global_batch_size)

# Setup trainer
trainer = train_lib.Trainer(
    config=config,
    task=task,
    train_ds=train_ds,
    validation_ds=validation_ds
)

# Run training
trainer.train()