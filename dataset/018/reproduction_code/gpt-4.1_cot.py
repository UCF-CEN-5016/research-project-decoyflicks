import subprocess
import sys

def reproduce_pip_use_feature_error():
    """
    Tries to run 'pip install --use-feature=2020-resolver .' 
    which triggers the error on pip >=20.3.
    """
    cmd = [sys.executable, '-m', 'pip', 'install', '--use-feature=2020-resolver', '.']
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Installation succeeded")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Installation failed with error:")
        print(e.stderr)

if __name__ == "__main__":
    reproduce_pip_use_feature_error()