import tensorflow as tf
from delf.python.training.model import delf_model

class DelfTest(tf.test.TestCase):
    def test_build_model(self):
        image_size = 321
        num_classes = 1000
        batch_size = 2
        input_shape = (batch_size, image_size, image_size, 3)

        model = delf_model.Delf(block3_strides=True, name='DELF')
        model.init_classifiers(num_classes)

        images = tf.random.uniform(input_shape, minval=-1.0, maxval=1.0, seed=0)
        blocks = {}

        desc_prelogits = model.backbone(images, intermediates_dict=blocks, training=False)
        desc_logits = model.desc_classification(desc_prelogits)
        self.assertAllEqual(desc_prelogits.shape, (batch_size, 2048))
        self.assertAllEqual(desc_logits.shape, (batch_size, num_classes))

        features = blocks['block3']
        attn_prelogits, _, _ = model.attention(features)
        attn_logits = model.attn_classification(attn_prelogits)
        self.assertAllEqual(attn_prelogits.shape, (batch_size, 1024))
        self.assertAllEqual(attn_logits.shape, (batch_size, num_classes))

if __name__ == '__main__':
    tf.test.main()