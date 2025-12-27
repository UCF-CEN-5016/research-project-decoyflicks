import subprocess
import sys

def main():
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--use-feature=2020-resolver', '.'])
    except subprocess.CalledProcessError as e:
        print(e)

if __name__ == "__main__":
    main()