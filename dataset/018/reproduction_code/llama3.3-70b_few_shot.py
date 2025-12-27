import subprocess
import sys

# Try to install using the invalid option
try:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--use-feature=2020-resolver', '.'])
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")