import tensorflow as tf
from official.vision import configs
from official.vision.modeling import factory
from official.vision.dataloaders import input_reader
from official.vision.dataloaders import maskrcnn_input
from official.vision.evaluation import evaluator

# 1. Minimal configuration (similar to tutorial)
params = configs.maskrcnn.MaskRCNN_50_FPN_1x_MS_COCO()
params.task.model.backbone.resnet.model_id = 18  # Smaller model for testing
params.task.train_data.global_batch_size = 1
params.task.validation_data.global_batch_size = 1

# 2. Create model and dummy dataset
model = factory.build_model(params)
dummy_image = tf.zeros((1, 512, 512, 3))
dummy_labels = {
    'boxes': tf.zeros((1, 100, 4)),
    'classes': tf.zeros((1, 100)),
    'masks': tf.zeros((1, 100, 28, 28))
}

# 3. Simulate validation (showing the bug)
@tf.function
def validation_step(inputs, model):
    outputs = model(inputs[0], training=False)
    # The key issue appears to be here - validation loss isn't properly computed
    loss = model.compute_loss(inputs[0], inputs[1], outputs)
    return loss

# 4. Run validation (will show 0.0 loss)
val_loss = validation_step((dummy_image, dummy_labels), model)
print(f"Validation loss: {val_loss.numpy()}")  # Will output 0.0

# 5. Compare with training loss (works fine)
with tf.GradientTape() as tape:
    outputs = model(dummy_image, training=True)
    train_loss = model.compute_loss(dummy_image, dummy_labels, outputs)
print(f"Training loss: {train_loss.numpy()}")  # Shows non-zero value