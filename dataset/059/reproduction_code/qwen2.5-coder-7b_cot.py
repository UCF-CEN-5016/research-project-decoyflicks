#!/usr/bin/env python3
"""
kcv_debug_helper.py

A small utility to assist debugging issues when fitting models with KerasCV.
It implements the checklist-style steps:
  1. Verify installation
  2. Check dependencies
  3. Review (parse) error messages
  4. Provide a minimal/simplified model setup
  5. Validate data handling (basic checks)
  6. Provide upgrade commands for libraries
  7. Run basic import/tests for keras-cv
  8. Prepare a compact issue report payload to share with the community

This script is diagnostic and advisory: it does not modify your environment by default.
"""

from __future__ import annotations
import sys
import os
import platform
import subprocess
import json
import logging
import importlib
from typing import Dict, List, Optional, Tuple

try:
    # importlib.metadata is available in Python 3.8+
    from importlib.metadata import version as pkg_version, PackageNotFoundError
except Exception:
    # fallback for older environments with pkg_resources
    try:
        from pkg_resources import get_distribution as _get_dist, DistributionNotFound

        def pkg_version(pkg_name: str) -> str:
            return _get_dist(pkg_name).version
    except Exception:
        def pkg_version(pkg_name: str) -> str:
            raise PackageNotFoundError()


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("kcv_debug_helper")


def get_python_info() -> Dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
    }


def get_package_version(package_name: str) -> Optional[str]:
    """Return installed version for package_name or None if not installed."""
    try:
        return pkg_version(package_name)
    except Exception:
        try:
            # best-effort import fallback to check presence
            pkg = importlib.import_module(package_name)
            ver = getattr(pkg, "__version__", None)
            if ver:
                return str(ver)
        except Exception:
            return None
    return None


def verify_installation(required: Optional[Dict[str, str]] = None) -> Dict[str, Optional[str]]:
    """
    Check key packages (tensorflow, keras-cv) and return their installed versions.
    `required` is an optional dict of package->minimum_version for context (not enforced).
    """
    if required is None:
        required = {"tensorflow": "", "keras-cv": ""}

    results: Dict[str, Optional[str]] = {}
    for pkg_name in required.keys():
        installed = get_package_version(pkg_name)
        results[pkg_name] = installed
        if installed:
            logger.info("Detected %s version: %s", pkg_name, installed)
        else:
            logger.warning("%s not detected in the current environment.", pkg_name)
    return results


def check_dependencies(dependencies: Optional[List[str]] = None) -> Dict[str, bool]:
    """
    Check presence of commonly required dependencies for KerasCV usage (e.g. opencv).
    Returns a map dependency->is_available.
    """
    if dependencies is None:
        dependencies = ["opencv-python", "opencv", "PIL", "numpy"]

    availability: Dict[str, bool] = {}
    for dep in dependencies:
        available = bool(get_package_version(dep))
        availability[dep] = available
        if available:
            logger.info("Dependency available: %s", dep)
        else:
            logger.warning("Dependency missing or not importable: %s", dep)
    return availability


def summarize_environment(extra_packages: Optional[List[str]] = None) -> Dict[str, Optional[str]]:
    """
    Collect environment summary useful for an issue report.
    Not exhaustive; adds extra_packages if provided.
    """
    summary = get_python_info()
    core_pkgs = ["pip", "setuptools", "wheel"]
    for pkg in core_pkgs + (extra_packages or []):
        summary[pkg] = get_package_version(pkg)
    # Add tensorflow and keras-cv if present
    summary["tensorflow"] = get_package_version("tensorflow")
    summary["keras-cv"] = get_package_version("keras-cv")
    return summary


def review_error_message(error_text: str) -> Dict[str, str]:
    """
    Parse an error string and make a best-effort classification with advice.
    Returns a dict with keys: 'type', 'short_advice', 'details'.
    """
    txt = error_text or ""
    lowered = txt.lower()
    result = {"type": "unknown", "short_advice": "Inspect the stack trace.", "details": txt}

    if "importerror" in lowered or "cannot import" in lowered or "no module named" in lowered:
        result["type"] = "ImportError"
        result["short_advice"] = "Check package installation and PYTHONPATH; ensure required packages are installed."
    elif "valueerror" in lowered or "value error" in lowered:
        result["type"] = "ValueError"
        result["short_advice"] = "Verify tensor shapes and data types match model expectations."
    elif "outofmemory" in lowered or "oom" in lowered or "cuda out of memory" in lowered:
        result["type"] = "OOM"
        result["short_advice"] = "Reduce batch size, use smaller model, or ensure GPU memory is free."
    elif "attributeerror" in lowered or "object has no attribute" in lowered:
        result["type"] = "AttributeError"
        result["short_advice"] = "Check API usage and package versions for breaking changes."
    elif "typeerror" in lowered or "type error" in lowered:
        result["type"] = "TypeError"
        result["short_advice"] = "Ensure inputs are of expected types (e.g., tf.Tensor vs numpy.ndarray)."
    elif "permission" in lowered or "permission denied" in lowered:
        result["type"] = "PermissionError"
        result["short_advice"] = "Check file/directory permissions and paths."
    else:
        # Heuristic checks
        if "traceback" in lowered or "tensorflow" in lowered:
            result["short_advice"] = "Share full traceback; it often contains the root cause."
    return result


def minimal_model_snippet() -> str:
    """
    Return a minimal example snippet to test a simple Keras+KerasCV training loop.
    This is a plain text snippet the user can copy/paste.
    """
    snippet = """
# Minimal Keras + KerasCV training example (copy this into a script or notebook cell)
import tensorflow as tf
try:
    import keras_cv
except Exception:
    pass

# Create tiny dataset
x = tf.random.uniform((8, 32, 32, 3))
y = tf.random.uniform((8,), maxval=10, dtype=tf.int32)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32,32,3)),
    tf.keras.layers.Conv2D(8, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# A single epoch run to validate basic training plumbing
model.fit(x, y, batch_size=4, epochs=1)
"""
    return snippet.strip()


def try_build_and_run_minimal(allow_execution: bool = False) -> Tuple[bool, str]:
    """
    Optionally attempt to execute the minimal model snippet.
    By default, this does NOT execute heavy operations unless allow_execution=True.
    Returns (success_flag, message).
    """
    if not allow_execution:
        return False, "Execution disabled; to run, call with allow_execution=True."

    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:
        return False, f"TensorFlow import failed: {exc!r}"

    try:
        # tiny run: 1 step training to verify that model.fit path is functional
        x = tf.random.uniform((4, 16, 16, 3))
        y = tf.random.uniform((4,), maxval=2, dtype=tf.int32)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(16, 16, 3)),
            tf.keras.layers.Conv2D(4, 3, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2),
        ])

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        model.fit(x, y, batch_size=2, epochs=1, verbose=0)
    except Exception as exc:
        return False, f"Minimal model run failed: {exc!r}"

    return True, "Minimal model run completed successfully."


def validate_data_paths(paths: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Basic validation for dataset path(s). Returns a dict path->status message.
    """
    if not paths:
        return {}

    results: Dict[str, str] = {}
    for p in paths:
        if not p:
            results[p] = "Empty path provided."
            continue
        if os.path.exists(p):
            if os.path.isdir(p):
                results[p] = f"Directory exists ({len(os.listdir(p))} entries)."
            else:
                try:
                    size = os.path.getsize(p)
                    results[p] = f"File exists (size: {size} bytes)."
                except Exception as e:
                    results[p] = f"File exists (could not determine size: {e!r})."
        else:
            results[p] = "Path does not exist."
    return results


def suggest_upgrade_commands(packages: Optional[List[str]] = None) -> List[str]:
    """
    Provide safe pip upgrade commands for common packages. These are suggestions only.
    """
    if packages is None:
        packages = ["tensorflow", "keras-cv", "opencv-python"]

    cmds = []
    python_exec = sys.executable or "python"
    for pkg in packages:
        cmds.append(f"{python_exec} -m pip install --upgrade {pkg}")
    return cmds


def run_basic_keras_cv_checks() -> Dict[str, Optional[str]]:
    """
    Try to import keras_cv and collect some basic attributes for diagnostics.
    """
    info: Dict[str, Optional[str]] = {"import_ok": None, "keras_cv_version": None, "notes": None}
    try:
        keras_cv = importlib.import_module("keras_cv")
        info["import_ok"] = "yes"
        info["keras_cv_version"] = getattr(keras_cv, "__version__", None)
        info["notes"] = "Imported keras_cv successfully."
    except Exception as exc:
        info["import_ok"] = "no"
        info["notes"] = f"Import failed: {exc!r}"
    return info


def prepare_issue_report(error_text: Optional[str] = None, data_paths: Optional[List[str]] = None) -> Dict:
    """
    Build a compact report dict containing environment, package versions,
    parsed error info, dependency checks, and suggested commands.
    This can be copy/pasted into GitHub or forum issues.
    """
    report = {
        "environment": summarize_environment(extra_packages=["opencv-python", "numpy", "PIL"]),
        "dependency_check": check_dependencies(["opencv-python", "numpy", "PIL"]),
        "keras_cv": run_basic_keras_cv_checks(),
        "minimal_model_snippet": minimal_model_snippet(),
        "upgrade_commands": suggest_upgrade_commands(["tensorflow", "keras-cv", "opencv-python"]),
    }
    if error_text:
        report["error_summary"] = review_error_message(error_text)
    if data_paths:
        report["data_paths"] = validate_data_paths(data_paths)
    return report


def pretty_print_report(report: Dict) -> None:
    """Print the report as pretty JSON for easy copy/paste."""
    try:
        print(json.dumps(report, indent=2, sort_keys=False))
    except Exception:
        logger.info("Could not serialize report to JSON; printing raw dict.")
        print(report)


def main(
    error_message: Optional[str] = None,
    dataset_paths: Optional[List[str]] = None,
    run_minimal: bool = False,
) -> int:
    """
    High-level orchestrator function.
    - error_message: an optional error message string to analyze
    - dataset_paths: optional list of dataset paths to validate
    - run_minimal: if True, attempt a minimal model run (may import TF)
    Returns exit code (0 success-ish, non-zero on notable failures).
    """
    logger.info("Starting KerasCV debug helper.")
    installed = verify_installation({"tensorflow": "", "keras-cv": ""})
    deps = check_dependencies(["opencv-python", "numpy", "PIL"])

    if error_message:
        parsed = review_error_message(error_message)
        logger.info("Error classification: %s", parsed.get("type"))
        logger.info("Short advice: %s", parsed.get("short_advice"))

    if dataset_paths:
        dp_results = validate_data_paths(dataset_paths)
        for p, status in dp_results.items():
            logger.info("Path: %s -> %s", p, status)

    snippet = minimal_model_snippet()
    logger.info("Provided a minimal model snippet to test basic training plumbing.")

    if run_minimal:
        logger.info("Attempting to run a minimal model to validate training path...")
        success, message = try_build_and_run_minimal(allow_execution=True)
        if success:
            logger.info(message)
        else:
            logger.error(message)

    report = prepare_issue_report(error_text=error_message, data_paths=dataset_paths)
    logger.info("Prepared diagnostic report. You can copy the JSON output for community help.")
    pretty_print_report(report)

    # Suggest next steps in logs
    logger.info("Suggested upgrade commands (run in your environment if desired):")
    for cmd in report.get("upgrade_commands", []):
        logger.info("  %s", cmd)

    # Exit code heuristic: return 0 unless we detected missing TF or keras-cv
    if not installed.get("tensorflow") or not installed.get("keras-cv"):
        logger.warning("tensorflow or keras-cv not detected; consider installing/upgrading before retrying.")
        return 2

    return 0


if __name__ == "__main__":
    # Example usage:
    # python kcv_debug_helper.py "Traceback (most recent call last): ..." /path/to/data --run
    import argparse

    parser = argparse.ArgumentParser(description="KerasCV debug helper utility.")
    parser.add_argument("--error", "-e", type=str, help="Paste the error message or traceback text.")
    parser.add_argument(
        "--paths",
        "-p",
        nargs="*",
        help="One or more dataset file or directory paths to validate.",
    )
    parser.add_argument(
        "--run-minimal",
        "-r",
        action="store_true",
        help="Attempt to execute a minimal model training run (will import TensorFlow).",
    )

    args = parser.parse_args()
    exit_code = main(error_message=args.error, dataset_paths=args.paths, run_minimal=args.run_minimal)
    sys.exit(exit_code)