import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_text  # Ensure TensorFlow Text is installed

print(tf.version.VERSION)

batch_size = 4
height, width = 512, 512
num_anchors = 9

score_outputs = tf.random.uniform((batch_size, height, width, num_anchors))
labels = tf.random.uniform((batch_size, height, width, num_anchors), minval=-1, maxval=2, dtype=tf.int32)

class RpnScoreLoss(object):
    def __init__(self, rpn_batch_size_per_im):
        self._rpn_batch_size_per_im = rpn_batch_size_per_im
        self._binary_crossentropy = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM, from_logits=True)

    def __call__(self, score_outputs, labels):
        with tf.name_scope('rpn_loss'):
            levels = sorted(score_outputs.keys())
            score_losses = []
            for level in levels:
                score_losses.append(
                    self._rpn_score_loss(
                        score_outputs[level],
                        labels[level],
                        normalizer=tf.cast(
                            tf.shape(score_outputs[level])[0] *
                            self._rpn_batch_size_per_im,
                            dtype=score_outputs[level].dtype)))

            return tf.math.add_n(score_losses)

    def _rpn_score_loss(self, score_outputs, score_targets, normalizer=1.0):
        with tf.name_scope('rpn_score_loss'):
            mask = tf.math.logical_or(tf.math.equal(score_targets, 1),
                                      tf.math.equal(score_targets, 0))
            score_targets = tf.math.maximum(score_targets,
                                            tf.zeros_like(score_targets))
            score_targets = tf.expand_dims(score_targets, axis=-1)
            score_outputs = tf.expand_dims(score_outputs, axis=-1)
            score_loss = self._binary_crossentropy(
                score_targets, score_outputs, sample_weight=mask)
            score_loss /= normalizer
            return score_loss

rpn_loss = RpnScoreLoss(rpn_batch_size_per_im=4)
try:
    output = rpn_loss(score_outputs, labels)
except ImportError as e:
    print(e)
    assert "undefined symbol: _ZN4absl12lts_2022062320raw_logging_internal21internal_log_functionB5cxx11E" in str(e)