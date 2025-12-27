import subprocess
import sys

# Attempt to use deprecated feature flag
command = [
    sys.executable,
    "-m",
    "pip",
    "install",
    "--use-feature=2020-resolver",
    "."
]

try:
    # This will raise CalledProcessError
    subprocess.run(command, check=True, capture_output=True, text=True)
except subprocess.CalledProcessError as e:
    print("Error occurred:")
    print(e.stderr)
    print(f"Current pip version: {subprocess.run([sys.executable, '-m', 'pip', '--version'], capture_output=True, text=True).stdout}")