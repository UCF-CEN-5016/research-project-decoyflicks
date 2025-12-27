import subprocess
import sys

# This will fail on pip >= 21.0
try:
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--use-feature=2020-resolver",
        "."
    ])
except subprocess.CalledProcessError as e:
    print(f"Error occurred: {e}")
    print("This is expected behavior in pip >= 21.0")

python -m pip install .

python -m pip install --upgrade pip
python -m pip install .