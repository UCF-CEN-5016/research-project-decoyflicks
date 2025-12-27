import tensorflow as tf
from official.vision.beta.configs import image_classification as img_cls_config
from official.vision.beta.tasks import image_classification

def create_dummy_dataset(batch_size=2, image_shape=[160, 160, 3], num_classes=1001):
    images = tf.random.normal([batch_size] + image_shape, dtype=tf.float16)
    labels = tf.one_hot(tf.random.uniform([batch_size], 0, num_classes, dtype=tf.int32), num_classes)
    return tf.data.Dataset.from_tensors((images, labels)).repeat()

def compute_loss(task, model, images, labels):
    with tf.GradientTape() as tape:
        outputs = model(images, training=True)
        loss = task.build_losses(labels=labels, model_outputs=outputs, aux_losses=model.losses)
    return loss, tape

config = img_cls_config.ImageClassificationTask(
    model=img_cls_config.ImageClassificationModel(
        num_classes=1001,
        input_size=[160, 160, 3],
        backbone=img_cls_config.Backbone(type='resnet', resnet=img_cls_config.ResNet(
            model_id=50,
            depth_multiplier=1.0,
            stem_type='v1',
            replace_stem_max_pool=True,
            resnetd_shortcut=True,
            se_ratio=0.25,
            stochastic_depth_drop_rate=0.0
        )),
        norm_activation=img_cls_config.common.NormActivation(
            activation='swish',
            norm_momentum=0.0,
            norm_epsilon=1e-05,
            use_sync_bn=False
        ),
        dropout_rate=0.25
    ),
    losses=img_cls_config.Losses(
        l2_weight_decay=4e-05,
        label_smoothing=0.1,
        one_hot=True
    )
)

task = image_classification.ImageClassificationTask(config)
model = task.build_model()
optimizer = tf.keras.optimizers.SGD(learning_rate=1.6, momentum=0.9)

dummy_dataset = create_dummy_dataset()

for step, (images, labels) in enumerate(dummy_dataset.take(10)):
    loss, tape = compute_loss(task, model, images, labels)
    
    print(f"Step {step}: Loss = {loss.numpy()}")
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    if tf.math.is_nan(loss):
        print("Loss became NaN - stopping training")
        break