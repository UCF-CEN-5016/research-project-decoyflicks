import tensorflow as tf
from official.vision.beta.tasks import image_classification
from official.vision.beta.configs import image_classification as classification_cfg

# Setup mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Minimal config matching bug report
config = classification_cfg.ImageClassificationTask(
    model=classification_cfg.ImageClassificationModel(
        num_classes=1001,
        input_size=[160, 160, 3],
        backbone=classification_cfg.Backbone(
            type='resnet',
            resnet=classification_cfg.ResNet(
                model_id=50,
                depth_multiplier=1.0,
                replace_stem_max_pool=True,
                resnetd_shortcut=True,
                se_ratio=0.25,
                stem_type='v1',
                stochastic_depth_drop_rate=0.0
            )
        ),
        norm_activation=classification_cfg.common.NormActivation(
            activation='swish',
            norm_epsilon=1e-5,
            norm_momentum=0.0,
            use_sync_bn=False
        ),
        dropout_rate=0.25
    ),
    losses=classification_cfg.Losses(
        l2_weight_decay=4e-5,
        label_smoothing=0.1,
        one_hot=True
    )
)

# Create task and model
task = image_classification.ImageClassificationTask(config)
model = task.build_model()

# Dummy input matching config
inputs = {
    'image': tf.random.uniform([2, 160, 160, 3], dtype=tf.float16),
    'label': tf.one_hot(tf.random.uniform([2], maxval=1001, dtype=tf.int32), 1001)
}

# Training step with high LR (matches bug report)
optimizer = tf.keras.optimizers.SGD(
    learning_rate=1.6,  # Very high LR from config
    momentum=0.9
)

# Run training steps until NaN appears
for step in range(100):
    with tf.GradientTape() as tape:
        outputs = task.train_step(inputs, model, optimizer, metrics=None)
        loss = outputs['loss']
    
    print(f"Step {step}: Loss = {loss.numpy()}")
    
    if tf.math.is_nan(loss):
        print("NaN loss detected!")
        break