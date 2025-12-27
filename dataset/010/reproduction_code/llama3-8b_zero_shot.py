import tensorflow as tf
from official.model_garden import metrics
from official.vision.configs.maskest import maskrcnn_mask_config
from official.vision.ops import preprocessed_op_builder

class MaskRCNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(MaskRCNN, self).__init__()
        self.backbone = tf.keras.applications.MobileNetV2(
            include_top=False,
            input_shape=(224, 224, 3)
        )
        self.head_mask = tf.keras.Sequential([
            tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes)
        ])
        self.head_bbox = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes)
        ])

    def call(self, inputs):
        x = self.backbone(inputs)
        mask_heads = tf.nn.relu(self.head_mask(x))
        bbox_heads = tf.nn.relu(self.head_bbox(x))
        return [mask_heads, bbox_heads]

num_classes = 10
model = MaskRCNN(num_classes)

@tf.function
def train_step(image, label):
    with tf.GradientTape() as tape:
        outputs = model(image)
        mask_loss = metrics.iou_loss(outputs[0], label)
        bbox_loss = metrics.binary_crossentropy(label, outputs[1])
        total_loss = mask_loss + bbox_loss
    gradients = tape.gradient(total_loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return {
        'mask_loss': mask_loss,
        'bbox_loss': bbox_loss,
        'total_loss': total_loss,
        'validation_loss': tf.zeros_like(mask_loss),
    }

image = tf.random.normal((1, 224, 224, 3))
label = tf.random.normal((1, 224, 224, num_classes))

for _ in range(10):
    results = train_step(image, label)
    print(results)