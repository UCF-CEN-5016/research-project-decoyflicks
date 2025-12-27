import logging
import os
import subprocess
import sys
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def change_directory(target_dir: Path) -> None:
    """Change the current working directory to target_dir."""
    logging.info("Changing working directory to: %s", target_dir)
    os.chdir(target_dir)


def ensure_package_directory(package_dir: Path) -> None:
    """Create a package directory and an __init__.py file inside it."""
    if not package_dir.exists():
        logging.info("Creating directory: %s", package_dir)
        package_dir.mkdir(parents=True, exist_ok=True)
    init_file = package_dir / "__init__.py"
    if not init_file.exists():
        logging.info("Creating file: %s", init_file)
        init_file.touch()


def run_script(script_path: Path, config: str, list_images: str, output: str) -> None:
    """Run the extract_features.py script with the provided arguments."""
    cmd = [
        sys.executable,
        str(script_path),
        "--config_path",
        config,
        "--list_images_path",
        list_images,
        "--output_dir",
        output,
    ]
    logging.info("Running command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    examples_dir = Path("/content/models/research/delf/delf/python/examples")
    datasets_dir = examples_dir / "datasets"
    script = examples_dir / "extract_features.py"
    config_file = "delf_config_example.pbtxt"
    images_list_file = "list_images.txt"
    output_directory = "data/oxford5k_features"

    change_directory(examples_dir)
    ensure_package_directory(datasets_dir)
    run_script(script, config_file, images_list_file, output_directory)


if __name__ == "__main__":
    main()