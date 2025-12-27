import subprocess
import sys

# Attempt to run the pip install command with deprecated --use-feature option
try:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--use-feature=2020-resolver", "."],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
except subprocess.CalledProcessError as e:
    print("Error output from pip install:")
    print(e.stderr)