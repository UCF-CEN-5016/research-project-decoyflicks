import subprocess

# Attempt to install with invalid option
try:
    result = subprocess.run(['python', '-m', 'pip', 'install', '--use-feature=2020-resolver', '.'], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")

print("Result:", result)