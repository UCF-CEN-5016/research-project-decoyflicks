import tensorflow as tf
from official.vision import registry_imports
from official.vision.beta.projects.panoptic_maskrcnn.configs import panoptic_maskrcnn as exp_cfg
from official.vision.beta.projects.panoptic_maskrcnn.tasks import panoptic_maskrcnn
from official.vision.beta.projects.panoptic_maskrcnn.dataloaders import input_reader
from official.core import exp_factory
from official.modeling import optimization

def get_config():
    config = exp_cfg.PanopticMaskRCNN()
    config.task.model.backbone.resnet.model_id = 50
    config.task.model.num_classes = 2
    config.task.train_data.global_batch_size = 1
    config.task.validation_data.global_batch_size = 1
    config.task.train_data.input_path = ''
    config.task.validation_data.input_path = ''
    return config

def fake_dataset(*args, **kwargs):
    image = tf.zeros((512, 512, 3), dtype=tf.float32)
    boxes = tf.constant([[0.1, 0.1, 0.5, 0.5]], dtype=tf.float32)
    masks = tf.zeros((1, 512, 512), dtype=tf.float32)
    classes = tf.constant([1], dtype=tf.int32)
    return tf.data.Dataset.from_tensors((image, {'gt_boxes': boxes, 'gt_classes': classes, 'gt_masks': masks}))

def main():
    config = get_config()
    model_dir = '/tmp/test_model'
    trainer = panoptic_maskrcnn.PanopticMaskRCNNTask(config.task)
    trainer._build_model()
    optimizer = optimization.OptimizerFactory(config.trainer.optimizer_config).build()
    trainer.initialize(optimizer)
    train_dataset = fake_dataset()
    validation_dataset = fake_dataset()
    trainer.train(train_dataset, validation_dataset=validation_dataset, model_dir=model_dir)

if __name__ == '__main__':
    registry_imports.import_registry()
    main()