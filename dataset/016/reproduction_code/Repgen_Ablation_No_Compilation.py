import os
import subprocess

# Set up environment variables
os.environ["PYTHONPATH"] = "/home/cdacapp1/scaling/models-master"
os.environ["DATA_DIR"] = "/scratch/cdacapp1/sowmya/actual_imagenet/tf_records"
os.environ["MODEL_DIR"] = "/home/cdacapp1/scaling/models-master/official/legacy/image_classification/checkpointss"

# IP addresses of nodes
ip1 = "192.168.1.100"
ip2 = "192.168.1.101"

# Create gpu.yaml and gpu1.yaml files
gpu_yaml = f"runtime:\n  distribution_strategy: 'multi_worker_mirrored'\n  worker_hosts: '{ip1}:6655,{ip2}:6655' \n  num_gpus: 2\n  task_index: 0"
gpu1_yaml = f"runtime:\n  distribution_strategy: 'multi_worker_mirrored'\n  worker_hosts: '{ip1}:6655,{ip2}:6655' \n  num_gpus: 2\n  task_index: 1"

with open("gpu.yaml", "w") as f:
    f.write(gpu_yaml)

with open("gpu1.yaml", "w") as f:
    f.write(gpu1_yaml)

# Execute training and evaluation on both nodes
subprocess.run(["python3", "classifier_trainer.py", "--mode=train_and_eval", "--model_type=resnet", "--dataset=imagenet", f"--model_dir={os.environ['MODEL_DIR']}", f"--data_dir={os.environ['DATA_DIR']}", "--config_file=gpu.yaml"], check=True)
subprocess.run(["python3", "classifier_trainer.py", "--mode=train_and_eval", "--model_type=resnet", "--dataset=imagenet", f"--model_dir={os.environ['MODEL_DIR']}", f"--data_dir={os.environ['DATA_DIR']}", "--config_file=gpu1.yaml"], check=True)