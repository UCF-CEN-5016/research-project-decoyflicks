import subprocess

def attempt_install():
    try:
        # Attempt to install using the --use-feature=2020-resolver option
        subprocess.run(['python', '-m', 'pip', 'install', '--use-feature=2020-resolver', '.'])
    except subprocess.CalledProcessError as e:
        print(f"Installation failed with error: {e}")

if __name__ == "__main__":
    attempt_install()