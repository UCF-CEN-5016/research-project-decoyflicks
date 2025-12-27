import tensorflow as tf
import tensorflow_datasets as tfds
from official.vision.modeling.maskrcnn_model import MaskRCNNModel

tf.random.set_seed(42)

batch_size = 4
image_size = (640, 640)

train_dataset = tfds.load('coco/2017', split='train', as_supervised=True)
train_dataset = train_dataset.map(lambda img, lbl: (tf.image.resize(img, image_size) / 255.0, lbl))
train_dataset = train_dataset.batch(batch_size)

model = MaskRCNNModel(
    backbone=tf.keras.applications.ResNet50(include_top=False, input_shape=(640, 640, 3)),
    decoder=None,
    rpn_head=None,
    detection_head=None,
    roi_generator=None,
    roi_sampler=None,
    roi_aligner=None,
    detection_generator=None,
    mask_head=None,
    mask_sampler=None,
    mask_roi_aligner=None,
    class_agnostic_bbox_pred=False,
    outer_boxes_scale=1.0
)

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy')

model.fit(train_dataset, epochs=5, validation_data=train_dataset)

validation_metrics = model.evaluate(train_dataset)
assert validation_metrics['validation_loss'] == 0.0
print(validation_metrics)