import tensorflow as tf
from delf.python.training.model import delg_model
from functools import partial
import math

class Delg(delg_model.Delf):
    def __init__(self,
                 block3_strides=True,
                 name='DELG',
                 gem_power=3.0,
                 embedding_layer_dim=2048,
                 scale_factor_init=45.25,
                 arcface_margin=0.1,
                 use_dim_reduction=False,
                 reduced_dimension=128,
                 dim_expand_channels=1024):
        super(Delg, self).__init__(block3_strides=block3_strides,
                                   name=name,
                                   pooling='gem',
                                   gem_power=gem_power,
                                   embedding_layer=True,
                                   embedding_layer_dim=embedding_layer_dim,
                                   use_dim_reduction=use_dim_reduction,
                                   reduced_dimension=reduced_dimension,
                                   dim_expand_channels=dim_expand_channels)
        self._embedding_layer_dim = embedding_layer_dim
        self._scale_factor_init = scale_factor_init
        self._arcface_margin = arcface_margin

    def init_classifiers(self, num_classes):
        super(Delg, self).init_classifiers(
            num_classes,
            desc_classification=self._create_backbone_classifier(num_classes))

    def _create_backbone_classifier(self, num_classes):
        self.cosine_weights = tf.Variable(
            initial_value=tf.initializers.GlorotUniform()(
                shape=[self._embedding_layer_dim, num_classes]),
            name='cosine_weights', trainable=True)
        self.scale_factor = tf.Variable(self._scale_factor_init,
                                        name='scale_factor', trainable=False)
        return partial(cosine_classifier_logits,
                                num_classes=num_classes,
                                cosine_weights=self.cosine_weights,
                                scale_factor=self.scale_factor,
                                arcface_margin=self._arcface_margin)

def cosine_classifier_logits(prelogits, labels, num_classes, cosine_weights, scale_factor, arcface_margin, training=True):
    normalized_prelogits = tf.math.l2_normalize(prelogits, axis=1)
    normalized_weights = tf.math.l2_normalize(cosine_weights, axis=0)
    cosine_sim = tf.matmul(normalized_prelogits, normalized_weights)

    if training and arcface_margin > 0.0:
        one_hot_labels = tf.one_hot(labels, num_classes)
        cosine_sim = apply_arcface_margin(cosine_sim, one_hot_labels, arcface_margin)

    logits = scale_factor * cosine_sim
    return logits

def apply_arcface_margin(cosine_sim, one_hot_labels, arcface_margin):
    theta = tf.acos(cosine_sim, name='acos')
    selected_labels = tf.where(tf.greater(theta, math.pi - arcface_margin),
                               tf.zeros_like(one_hot_labels), one_hot_labels, name='selected_labels')
    final_theta = tf.where(tf.cast(selected_labels, dtype=tf.bool),
                         theta + arcface_margin, theta, name='final_theta')
    return tf.cos(final_theta, name='cosine_sim_with_margin')

# Main script
if __name__ == "__main__":
    delg_model_instance = Delg()
    # Add code to set up the model for training and prepare data