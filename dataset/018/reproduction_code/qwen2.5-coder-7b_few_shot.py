import subprocess
from typing import List


def build_install_command(python_executable: str = "python",
                          package_path: str = ".",
                          use_new_resolver: bool = True) -> List[str]:
    command = [python_executable, "-m", "pip", "install"]
    if use_new_resolver:
        command.append("--use-feature=2020-resolver")
    command.append(package_path)
    return command


def run_command(cmd: List[str]) -> None:
    subprocess.run(cmd)


def main() -> None:
    install_cmd = build_install_command()
    run_command(install_cmd)


if __name__ == "__main__":
    main()