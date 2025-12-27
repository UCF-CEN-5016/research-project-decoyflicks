import os
import subprocess
import sys

def main():
    # Step 1: Check Python and pip versions
    print("Python version:", sys.version)
    print("pip version:", subprocess.check_output([sys.executable, '-m', 'pip', '--version']).decode().strip())

    # Step 2: Upgrade pip if necessary
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])

    # Step 3: Clone TensorFlow models repository
    subprocess.check_call(['git', 'clone', 'https://github.com/tensorflow/models.git'])

    # Step 4: Navigate to object detection directory
    os.chdir('models/research/object_detection')

    # Step 5: Create a virtual environment
    subprocess.check_call([sys.executable, '-m', 'venv', 'tfod2_env'])

    # Step 6: Activate the virtual environment
    activate_script = os.path.join('tfod2_env', 'bin', 'activate')
    subprocess.call(['source', activate_script], shell=True)

    # Step 7: Install TensorFlow
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow'])

    # Step 8: Install required dependencies
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

    # Step 9: Attempt to install the local package
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--use-feature=2020-resolver', '.'])
    except subprocess.CalledProcessError as e:
        print("Error occurred:", e)

if __name__ == "__main__":
    main()