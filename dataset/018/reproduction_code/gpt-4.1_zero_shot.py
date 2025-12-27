import subprocess
import sys

subprocess.run([sys.executable, '-m', 'pip', 'install', '--use-feature=2020-resolver', '.'])