import os
import sys
import subprocess
import importlib
from pathlib import Path

def _install_editable_package(parent_path: str = '../') -> None:
    """Install a package in editable mode using pip."""
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--editable', parent_path], check=True)

def _import_all_from(module_name: str) -> None:
    """Import all public attributes from a module into the current global namespace."""
    module = importlib.import_module(module_name)
    for name, value in vars(module).items():
        if not name.startswith('_'):
            globals()[name] = value

def _change_working_directory(path: Path) -> None:
    """Change the current working directory to the given path."""
    os.chdir(path)

def main() -> None:
    target_dir = Path('examples') / 'mms' / 'tts'
    _change_working_directory(target_dir)
    _install_editable_package('../')
    _import_all_from('commons')

if __name__ == '__main__':
    main()