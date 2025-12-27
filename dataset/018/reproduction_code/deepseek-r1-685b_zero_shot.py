import subprocess
import sys

def test_pip_feature_resolver():
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--use-feature=2020-resolver", "."],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(e.stderr)

if __name__ == "__main__":
    test_pip_feature_resolver()