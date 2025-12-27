import subprocess

subprocess.run([
    'python', '-m', 'pip', 'install',
    '--use-feature=2020-resolver', '.'
])

import subprocess

# Attempt to install with --use-feature=2020-resolver, which may fail due to outdated pip
subprocess.run([
    'python', '-m', 'pip', 'install',
    '--use-feature=2020-resolver', '.'
])