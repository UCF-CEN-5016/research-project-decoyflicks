import os
import stat
import subprocess
from pathlib import Path

SCRIPT_NAME = "test_tune.sh"
TEST_DIR = Path("test")
SCRIPT_CONTENT = """#!/bin/bash
HF_PATH=../
NGPUS=6
python -m transformers.tuning run --output_dir output --best_model_path best_model --max_steps 10
""".strip() + "\n"


def ensure_test_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def change_to_directory(path: Path) -> None:
    os.chdir(path)


def install_current_package() -> None:
    subprocess.run(["pip", "install", "."], check=True)


def write_executable_script(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    current_mode = path.stat().st_mode
    path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def run_script(path: Path, arg: str = "tune") -> None:
    subprocess.run([str(path), arg])


def main() -> None:
    test_path = ensure_test_directory(TEST_DIR)
    change_to_directory(test_path)
    install_current_package()
    script_path = test_path / SCRIPT_NAME
    write_executable_script(script_path, SCRIPT_CONTENT)
    run_script(script_path, "tune")


if __name__ == "__main__":
    main()