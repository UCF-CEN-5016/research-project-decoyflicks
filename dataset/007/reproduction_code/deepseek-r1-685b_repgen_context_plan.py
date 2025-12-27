import os
import sys
import subprocess
import shutil

def run_command(command, cwd=None):
    """Executes a shell command and prints its output."""
    try:
        process = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, cwd=cwd)
        print(process.stdout)
        if process.stderr:
            print(process.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        sys.exit(1)

def setup_tensorflow_models(base_dir="."):
    """Clones the TensorFlow models repository and sets up the environment."""
    object_detection_path = os.path.join(base_dir, 'models', 'research', 'object_detection')

    if not os.path.exists(object_detection_path):
        print(f"Cloning TensorFlow models into {object_detection_path}...")
        run_command(f'git clone https://github.com/tensorflow/models.git {object_detection_path}')
    else:
        print(f"TensorFlow models already exist at {object_detection_path}. Skipping clone.")

    print("Installing required Python packages...")
    run_command('pip install tensorflow==2.10.1')
    run_command('pip install tensorflow-addons==0.20.0')

    config_file_name = 'ssd_efficientdet_d0_512x512_coco17_tpu-8.config'
    source_config_path = os.path.join(object_detection_path, config_file_name)
    destination_config_path = os.path.join(object_detection_path, config_file_name) # Copying to the same directory for simplicity as per original

    if os.path.exists(source_config_path):
        print(f"Copying configuration file {config_file_name}...")
        # Using shutil.copy for cross-platform file copying
        shutil.copy(source_config_path, destination_config_path)
    else:
        print(f"Configuration file {source_config_path} not found. Skipping copy.")

    config_util_file = os.path.join(object_detection_path, 'utils', 'config_util.py')
    print(f"Editing {config_util_file}...")
    try:
        with open(config_util_file, 'r+', encoding='latin-1') as f:
            content = f.read()
            f.seek(0)
            f.write(content)
            f.truncate()
        print("Successfully edited config_util.py.")
    except FileNotFoundError:
        print(f"Error: {config_util_file} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error editing config_util.py: {e}")
        sys.exit(1)

    print("Running model_main_tf2.py...")
    # Navigate to the object_detection directory to run the script
    run_command(
        f'python3.9 model_main_tf2.py --pipeline_config_path={config_file_name} --model_dir=training --alsologtostderr',
        cwd=object_detection_path
    )

if __name__ == "__main__":
    setup_tensorflow_models()
