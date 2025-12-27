import subprocess

def run_install_command():
    try:
        result = subprocess.run(['python', '-m', 'pip', 'install', '--use-feature=2020-resolver', '.'], check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(e.stderr)

if __name__ == '__main__':
    run_install_command()