import tensorflow as tf
from official.vision.beta.configs import image_classification as img_cls_config
from official.vision.beta.tasks import image_classification

# Define the model configuration
def build_model_config():
    backbone_config = img_cls_config.Backbone(
        type='resnet',
        resnet=img_cls_config.ResNet(
            model_id=50,
            depth_multiplier=1.0,
            stem_type='v1',
            replace_stem_max_pool=True,
            resnetd_shortcut=True,
            se_ratio=0.25,
            stochastic_depth_drop_rate=0.0
        )
    )
    norm_activation_config = img_cls_config.common.NormActivation(
        activation='swish',
        norm_momentum=0.0,
        norm_epsilon=1e-05,
        use_sync_bn=False
    )
    model_config = img_cls_config.ImageClassificationModel(
        num_classes=1001,
        input_size=[160, 160, 3],
        backbone=backbone_config,
        norm_activation=norm_activation_config,
        dropout_rate=0.25
    )
    return model_config

# Setup the minimal configuration matching the bug report
config = img_cls_config.ImageClassificationTask(
    model=build_model_config(),
    losses=img_cls_config.Losses(
        l2_weight_decay=4e-05,
        label_smoothing=0.1,
        one_hot=True
    )
)

# Create dummy dataset matching the config
def create_dummy_dataset(batch_size=2):
    images = tf.random.normal([batch_size, 160, 160, 3], dtype=tf.float16)
    labels = tf.one_hot(tf.random.uniform([batch_size], 0, 1000, dtype=tf.int32), 1001)
    return tf.data.Dataset.from_tensors((images, labels)).repeat()

# Create task, model, and optimizer
task = image_classification.ImageClassificationTask(config)
model = task.build_model()
optimizer = tf.keras.optimizers.SGD(learning_rate=1.6, momentum=0.9)

# Training loop
for step in range(10):
    with tf.GradientTape() as tape:
        images, labels = next(iter(create_dummy_dataset()))
        outputs = model(images, training=True)
        loss = task.build_losses(labels=labels, model_outputs=outputs, aux_losses=model.losses)
    
    print(f"Step {step}: Loss = {loss.numpy()}")
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    if tf.reduce_any(tf.math.is_nan(grad) for grad in grads):
        print("Loss became NaN - stopping training")
        break