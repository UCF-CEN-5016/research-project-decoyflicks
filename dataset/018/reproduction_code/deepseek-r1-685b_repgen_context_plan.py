import subprocess
import sys

def install_package():
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--use-feature=2020-resolver",
        "."
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Error occurred:")
        print(e.stderr)
        print(f"Current pip version: {get_pip_version()}")

def get_pip_version():
    result = subprocess.run([sys.executable, '-m', 'pip', '--version'], capture_output=True, text=True)
    return result.stdout

if __name__ == "__main__":
    install_package()