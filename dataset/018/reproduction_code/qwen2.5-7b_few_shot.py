import subprocess

# Command to install a package with optional flag '--use-feature=2020-resolver'
install_command = ['python', '-m', 'pip', 'install', '--use-feature=2020-resolver', '.']

# Execute the installation command
subprocess.run(install_command)