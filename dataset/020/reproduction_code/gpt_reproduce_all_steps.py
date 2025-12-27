import tensorflow as tf
import tfm.core.train_lib as train_lib

def main():
    distribution_strategy = tf.distribute.get_strategy()
    # Mock or minimal exp_config and task setup
    exp_config = {
        'train': {'batch_size': 2, 'steps_per_epoch': 1},
        'eval': {'batch_size': 2, 'steps': 1},
        # add minimal config fields required by your task
    }
    class DummyTask:
        def __init__(self):
            pass
    task = DummyTask()
    model_dir = '/tmp/model_dir'

    model, eval_logs = train_lib.run_experiment(
        distribution_strategy=distribution_strategy,
        task=task,
        mode='train_and_eval',
        params=exp_config,
        model_dir=model_dir,
        run_post_eval=True
    )

if __name__ == "__main__":
    main()