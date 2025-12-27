from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import subprocess
import sys
from typing import Iterable, List


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _build_pip_command(python_executable: str, packages: Iterable[str], user: bool = True, upgrade: bool = False) -> List[str]:
    cmd = [python_executable, "-m", "pip", "install"]
    if user:
        cmd.append("--user")
    if upgrade:
        cmd.append("--upgrade")
    cmd.extend(packages)
    return cmd


def _run_command(command: List[str]) -> None:
    logging.info("Running command: %s", " ".join(command))
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as exc:
        logging.error("Command failed with exit code %s: %s", exc.returncode, " ".join(command))
        raise


def upgrade_pip(python_executable: str) -> None:
    cmd = _build_pip_command(python_executable, ["pip"], user=True, upgrade=True)
    _run_command(cmd)


def install_packages(python_executable: str, packages: Iterable[str]) -> None:
    cmd = _build_pip_command(python_executable, list(packages), user=True, upgrade=False)
    _run_command(cmd)


def main() -> int:
    python_exec = sys.executable or "python"
    try:
        upgrade_pip(python_exec)
        install_packages(python_exec, ["tensorflow", "models"])
    except Exception:
        logging.exception("Installation failed.")
        return 1
    logging.info("All installations completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())