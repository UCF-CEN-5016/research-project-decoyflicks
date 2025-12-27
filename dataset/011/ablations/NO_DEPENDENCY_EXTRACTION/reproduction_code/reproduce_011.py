import os
import subprocess

# Dependencies: TensorFlow 2.6.0, TensorFlow Addons, TensorFlow Datasets, etc.

def setup_environment():
    subprocess.run(["docker", "pull", "nvcr.io/nvidia/tensorflow:21.12-tf2-py3"])
    subprocess.run(["docker", "run", "--gpus", "all", "-it", "nvcr.io/nvidia/tensorflow:21.12-tf2-py3", "/bin/bash"])

def install_dependencies():
    subprocess.run(["pip", "install", "gin-config==0.5.0", "sentencepiece==0.1.97", "seqeval==1.2.2", 
                    "pycocotools==2.0.7", "opencv-python==4.6.0.66", "sacrebleu==1.2.10", 
                    "jupyter-tensorboard==0.2.0", "tensorboard==2.6.0", "tensorboard-data-server==0.6.1", 
                    "tensorboard-plugin-wit==1.8.1", "tensorflow==2.6.3", "tensorflow-addons==0.14.0", 
                    "tensorflow-datasets==3.2.1", "tensorflow-estimator==2.6.0", "tensorflow-hub==0.12.0", 
                    "tensorflow-metadata==1.5.0", "tensorflow-model-optimization==0.7.3", 
                    "tensorflow-text==2.6.0"])

def clone_repository():
    subprocess.run(["git", "clone", "https://github.com/tensorflow/models.git"])
    os.chdir("models")
    subprocess.run(["git", "checkout", "r2.6.0"])

def run_training():
    os.chdir("official/vision/beta")
    os.environ["PYTHONPATH"] = os.path.realpath("../../../") + ":" + os.environ["PYTHONPATH"]
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    
    command = [
        "python", "train.py", "--experiment=resnet_rs_imagenet", "--mode=train_and_eval",
        "--model_dir=./results_resnet-rs50_tf2", 
        "--config_file=configs/experiments/image_classification/imagenet_resnetrs50_i160.yaml",
        "--params_override='runtime.enable_xla=False, runtime.num_gpus=1, runtime.mixed_precision_dtype=float16, "
        "runtime.distribution_strategy='one_device', task.train_data.input_path='/ppusw/datasets/vision/imagenet/tfrecords/train*', "
        "task.train_data.global_batch_size=2, task.train_data.dtype=float16, "
        "task.validation_data.input_path='/ppusw/datasets/vision/imagenet/tfrecords/valid*', "
        "task.validation_data.global_batch_size=2, task.validation_data.dtype=float16, "
        "trainer.train_steps=100, trainer.validation_steps=25000, trainer.validation_interval=640583, "
        "trainer.steps_per_loop=640583, trainer.summary_interval=640583, trainer.checkpoint_interval=640583, "
        "trainer.optimizer_config.ema='', trainer.optimizer_config.learning_rate.cosine.decay_steps=100, "
        "trainer.optimizer_config.warmup.linear.warmup_steps=0'"
    ]
    
    subprocess.run(command)

if __name__ == "__main__":
    setup_environment()
    install_dependencies()
    clone_repository()
    run_training()