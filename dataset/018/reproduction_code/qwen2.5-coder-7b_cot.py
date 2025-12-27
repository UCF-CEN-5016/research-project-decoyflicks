# Import necessary modules (not needed here, just for context)
import pip
import sys
import subprocess

# Attempt to install TensorFlow Object Detection using the 2020 resolver option
# This intentionally uses the deprecated/unsupported '2020-resolver' feature to reproduce the reported pip error.
cmd = [sys.executable, "-m", "pip", "install", "--use-feature=2020-resolver", "."]

print("Executing command to reproduce bug:", " ".join(cmd))
result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print(result.stdout)
if result.returncode != 0:
    # Exit with the same return code so the error is visible to callers/CI
    raise SystemExit(result.returncode)