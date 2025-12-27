import tensorflow as tf
from official.model_garden import metrics

def build_backbone(input_shape=(224, 224, 3)):
    return tf.keras.applications.MobileNetV2(include_top=False, input_shape=input_shape)

def build_head(filters, num_classes):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes)
    ])

class MaskRCNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(MaskRCNN, self).__init__()
        self.backbone = build_backbone()
        self.mask_head = build_head(256, num_classes)
        self.bbox_head = build_head(128, num_classes)

    def call(self, inputs):
        features = self.backbone(inputs)
        mask_logits = tf.nn.relu(self.mask_head(features))
        bbox_logits = tf.nn.relu(self.bbox_head(features))
        return [mask_logits, bbox_logits]

NUM_CLASSES = 10
model = MaskRCNN(NUM_CLASSES)

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        outputs = model(images)
        mask_loss = metrics.iou_loss(outputs[0], labels)
        bbox_loss = metrics.binary_crossentropy(labels, outputs[1])
        total_loss = mask_loss + bbox_loss
    gradients = tape.gradient(total_loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return {
        'mask_loss': mask_loss,
        'bbox_loss': bbox_loss,
        'total_loss': total_loss,
        'validation_loss': tf.zeros_like(mask_loss),
    }

sample_image = tf.random.normal((1, 224, 224, 3))
sample_label = tf.random.normal((1, 224, 224, NUM_CLASSES))

for _ in range(10):
    results = train_step(sample_image, sample_label)
    print(results)