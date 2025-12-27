import tensorflow as tf
from official.vision.beta.projects.maskrcnn import model as maskrcnn_model
from official.vision.beta.dataloaders import input_reader
from official.vision.beta.ops import box_ops

def get_fake_dataset():
    def gen():
        image = tf.zeros([640, 480, 3], tf.float32)
        boxes = tf.constant([[0.1, 0.1, 0.5, 0.5]], tf.float32)
        classes = tf.constant([1], tf.int32)
        masks = tf.zeros([1, 640, 480], tf.float32)
        yield {
            'image': image,
            'groundtruth_boxes': boxes,
            'groundtruth_classes': classes,
            'groundtruth_instance_masks': masks,
        }
    return tf.data.Dataset.from_generator(
        gen,
        output_signature={
            'image': tf.TensorSpec([640, 480, 3], tf.float32),
            'groundtruth_boxes': tf.TensorSpec([None, 4], tf.float32),
            'groundtruth_classes': tf.TensorSpec([None], tf.int32),
            'groundtruth_instance_masks': tf.TensorSpec([None, 640, 480], tf.float32),
        }
    ).batch(1)

def create_model():
    params = maskrcnn_model.MaskRCNN.default_hparams()
    params.model.backbone.type = 'resnet50'
    params.model.num_classes = 2
    return maskrcnn_model.MaskRCNN(params)

train_ds = get_fake_dataset()
val_ds = get_fake_dataset()

model = create_model()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        outputs = model(batch, training=True)
        loss = outputs['loss']
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

@tf.function
def val_step(batch):
    outputs = model(batch, training=False)
    loss = outputs['loss']
    return loss

for step, batch in enumerate(train_ds.take(5)):
    loss = train_step(batch)
    if step % 2 == 0:
        val_loss = 0.0
        for val_batch in val_ds.take(1):
            val_loss = val_step(val_batch)
        print(f"step {step}, train_loss: {loss.numpy()}, validation_loss: {val_loss.numpy()}")