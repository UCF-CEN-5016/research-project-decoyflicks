# This minimal script reproduces the bug by running 'tune' without any additional parameters beyond what's necessary
import subprocess

def run_tune():
    # Properly quote and pass arguments to avoid shell misinterpretation
    subprocess.run(['./test_tune.sh', 'tune'])

if __name__ == "__main__":
    run_tune()