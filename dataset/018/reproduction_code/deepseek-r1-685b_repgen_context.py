import subprocess
import sys

def run_command(command):
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Error occurred:")
        print(e.stderr)

def get_pip_version():
    return run_command([sys.executable, '-m', 'pip', '--version'])

def install_package_with_feature_flag():
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--use-feature=2020-resolver",
        "."
    ]
    return run_command(command)

if __name__ == "__main__":
    install_result = install_package_with_feature_flag()
    print(f"Current pip version: {get_pip_version()}")