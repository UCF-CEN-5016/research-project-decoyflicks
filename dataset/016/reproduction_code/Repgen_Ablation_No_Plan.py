import os
import json
import tensorflow as tf
from absl import flags
from official import DatasetBuilder, build_learning_rate, build_optimizer, build_stats, configure_cluster, configure_optimizer, get_callbacks, get_config, get_distribution_strategy, get_strategy_scope, initialize_common_flags, override_params_dict, save_params_dict_to_yaml, set_gpu_thread_mode_and_count, set_mixed_precision_policy, set_session_config
import utils
import tfexample_utils

FLAGS = flags.FLAGS

def test_train_and_eval():
    saved_flag_values = flags.get_flag_values()
    train_lib.tfm_flags.define_flags()
    FLAGS.mode = 'train'
    FLAGS.model_dir = os.path.join(os.getcwd(), 'model_dir')
    if not tf.io.gfile.exists(FLAGS.model_dir):
        tf.io.gfile.makedirs(FLAGS.model_dir)
    
    data_dir = os.path.join(os.getcwd(), 'data')
    if not tf.io.gfile.exists(data_dir):
        tf.io.gfile.makedirs(data_dir)
    
    self._data_path = os.path.join(data_dir, 'data.tfrecord')
    examples = [utils.make_yt8m_example() for _ in range(8)]
    tfexample_utils.dump_to_tfrecord(self._data_path, tf_examples=examples)

    average_precision = {'top_k': 20} if FLAGS.use_average_precision_metric else None
    params_override = json.dumps({
        'runtime': {
            'distribution_strategy': 'mirrored',
            'mixed_precision_dtype': 'float32',
        },
        'trainer': {
            'train_steps': 2,
            'validation_steps': 2,
        },
        'task': {
            'model': {
                'backbone': {
                    'type': 'dbof',
                    'dbof': {
                        'cluster_size': 16,
                        'hidden_size': 16,
                        'use_context_gate_cluster_layer': True,
                    },
                },
                'head': {
                    'type': 'moe',
                    'moe': {
                        'use_input_context_gate': True,
                        'use_output_context_gate': True,
                    },
                },
            },
            'train_data': {
                'input_path': self._data_path,
                'global_batch_size': 4,
            },
            'validation_data': {
                'input_path': self._data_path,
                'segment_labels': FLAGS.use_segment_level_labels,
                'global_batch_size': 4,
            },
            'evaluation': {
                'average_precision': average_precision,
            },
        },
    })
    override_params_dict(params_override)

    with get_distribution_strategy().scope():
        model = DatasetBuilder.build_model()
        optimizer = build_optimizer(learning_rate=build_learning_rate())
        train_dataset = build_stats.Dataset.from_tfrecord(self._data_path)
        val_dataset = build_stats.Dataset.from_tfrecord(self._data_path, is_training=False)
        callbacks = get_callbacks()

        train_lib.train.main(
            model=model,
            optimizer=optimizer,
            train_dataset=train_dataset,
            validation_dataset=val_dataset,
            callbacks=callbacks,
        )

    FLAGS.set_flags_from_values(saved_flag_values)

if __name__ == '__main__':
    flags.mark_flag_as_required('mode')
    flags.run(test_train_and_eval)